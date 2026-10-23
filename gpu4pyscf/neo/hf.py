import cupy
import numpy
from scipy.special import erf
from pyscf import gto, neo
from pyscf import lib as pyscf_lib
from pyscf.data import nist
from pyscf.neo import hf as hf_cpu
from pyscf.scf import chkfile
from gpu4pyscf import __config__
from gpu4pyscf import scf as scf_gpu
from gpu4pyscf import lib
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import asarray, get_avail_mem, tag_array
from gpu4pyscf.qmmm.itrf import _mm_charge_integrals


WITH_META_LOWDIN = getattr(__config__, 'scf_analyze_with_meta_lowdin', True)


# NEO stores SCF data in nested dict/list structures.  The generic GPU4PySCF
# converters only convert top-level attributes, so conversion at the NEO
# CPU/GPU boundary needs to recurse through these containers.
def _to_cpu(x):
    if isinstance(x, cupy.ndarray):
        return x.get()
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(_to_cpu(v) for v in x)
    if isinstance(x, list):
        return [_to_cpu(v) for v in x]
    return x


def _to_gpu(x):
    if isinstance(x, numpy.ndarray):
        return cupy.asarray(x)
    if isinstance(x, dict):
        return {k: _to_gpu(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(_to_gpu(v) for v in x)
    if isinstance(x, list):
        return [_to_gpu(v) for v in x]
    return x


def _grouped_hcore(components, mols, int1e_opt=None):
    from gpu4pyscf.gto.int3c1e import int1e_grids
    from gpu4pyscf.neo import int1e, j_engine_3c2e

    component_names = list(components)
    for mol in mols.values():
        assert not mol.nucmod
    # Each nuclear component uses the same classical centers but an
    # independent AO basis.  Build their classical-nucleus potential
    # integrals in one component-local 3c2e job.
    nucmol = gto.mole.fakemol_for_charges(next(iter(mols.values())).atom_coords())
    int3c2e_opt = j_engine_3c2e.Int3c2eOpt(mols, nucmol).build()
    atom_charges = {t: mols[t].atom_charges() for t in component_names}
    if all(numpy.array_equal(atom_charges[component_names[0]], atom_charges[t])
           for t in component_names[1:]):
        # The common charge vector has one contraction result per component
        # because the AO-pair blocks remain component-local.
        auxvec = cupy.asarray(-atom_charges[component_names[0]], dtype=numpy.float64)
        auxvec = int3c2e_opt.auxmol.apply_C_dot(auxvec, axis=-1)
        vext = int3c2e_opt.contract_auxvec(auxvec)
    else:
        # Initial guesses treat the other quantum nuclei as classical, so
        # each component has a different charge vector on the same centers.
        auxvec = {
            t: int3c2e_opt.auxmol.apply_C_dot(
                cupy.asarray(-atom_charges[t], dtype=numpy.float64), axis=-1)
            for t in component_names}
        vext = int3c2e_opt.contract_auxvec(auxvec, componentwise=True)

    if int1e_opt is None:
        int1e_opt = int1e.Int1eOpt(mols)
    kinetic = int1e_opt.get_kin()
    hcore = {}
    for t in component_names:
        comp = components[t]
        mol = mols[t]
        # Apply the original C^T V C output transformation per component.
        h = int3c2e_opt.component_opts[t].mol.apply_CT_mat_C(vext[t])
        # Each nucleus has its own charge and mass: q*V + T/m.
        h *= comp.charge
        h += kinetic[t] / comp.mass
        mm_mol = None
        if hasattr(mol, 'super_mol'):
            mm_mol = mol.super_mol.mm_mol
        elif hasattr(mol, 'mm_mol'):
            mm_mol = mol.mm_mol
        if mm_mol is not None:
            h -= _mm_charge_integrals(mm_mol, mol, int1e_grids) * comp.charge
        hcore[t] = h
    return hcore


def _transform_by_cholesky(h, chol):
    '''Transform a matrix batch to L^-1 h L^-H, with chol = L.'''
    h_orth = cupy.linalg.solve(chol, h)
    h_orth = cupy.linalg.solve(chol, h_orth.swapaxes(-1, -2).conj()).swapaxes(-1, -2).conj()
    return h_orth


def _eig_batch(h, s=None, x=None):
    if x is None:
        if h.dtype != s.dtype:
            s = s.astype(h.dtype)
        chol = cupy.linalg.cholesky(s)
        if chol.ndim < h.ndim:
            chol = cupy.broadcast_to(chol, h.shape)
        h_orth = _transform_by_cholesky(h, chol)
        energy, coeff_orth = cupy.linalg.eigh(h_orth)
        coeff = cupy.linalg.solve(chol.swapaxes(-1, -2).conj(), coeff_orth)
    else:
        h_orth = x.swapaxes(-1, -2).conj() @ h @ x
        energy, coeff_orth = cupy.linalg.eigh(h_orth)
        coeff = x @ coeff_orth
    return energy, coeff


def _grouped_eig(components, h, s, overwrite=False, x=None):
    energy = {}
    coeff = {}
    groups = {}
    # Batched eigh requires equal matrix shapes, dtypes, and either an
    # orthogonalizer for every component in the group or none of them.
    for t in components:
        x_t = None if x is None else x[t]
        x_key = None if x_t is None else (x_t.shape, x_t.dtype)
        key = (h[t].shape, h[t].dtype, x_key)
        groups.setdefault(key, []).append(t)
    for keys in groups.values():
        matrices = s if x is None or x[keys[0]] is None else x
        shared = {}
        for t in keys:
            shared.setdefault(matrices[t].data.ptr, []).append(t)
        eig_groups = [(group, True) for group in shared.values() if len(group) > 1]
        unique = [group[0] for group in shared.values() if len(group) == 1]
        if unique:
            eig_groups.append((unique, False))
        for eig_keys, share_matrix in eig_groups:
            h_batch = cupy.stack([h[t] for t in eig_keys])
            matrix = matrices[eig_keys[0]] if share_matrix else \
                cupy.stack([matrices[t] for t in eig_keys])
            if x is None or x[eig_keys[0]] is None:
                # SCF.eig calls a two-dimensional generalized eigensolver.
                energy_batch, coeff_batch = _eig_batch(h_batch, matrix)
            else:
                # This is SCF.eig's x^H h x branch with a component batch axis.
                energy_batch, coeff_batch = _eig_batch(h_batch, x=matrix)
            for i, t in enumerate(eig_keys):
                energy[t] = energy_batch[i]
                coeff[t] = coeff_batch[i]
    return energy, coeff


def _grouped_occ(components, mo_energy):
    mo_occ = {}
    groups = {}
    # Components with the same orbital-vector layout share argsort and scatter
    # operations while retaining their own selected nuclear state.
    for t in components:
        key = (mo_energy[t].shape, mo_energy[t].dtype)
        groups.setdefault(key, []).append(t)
    for keys in groups.values():
        energy = cupy.stack([mo_energy[t] for t in keys])
        order = cupy.argsort(energy, axis=1)
        occ = cupy.zeros_like(energy)
        rows = cupy.arange(len(keys))
        states = cupy.asarray([components[t].nuc_occ_state for t in keys])
        # Unlike the electronic get_occ, each distinguishable nucleus occupies
        # nuc_occ_state with its component's nnuc occupation.
        occ[rows, order[rows, states]] = cupy.asarray([components[t].mol.nnuc for t in keys])
        if any(components[t].verbose >= logger.INFO for t in keys):
            frontier = energy[rows[:,None], order[:,:2]].get()
            for i, t in enumerate(keys):
                comp = components[t]
                if comp.verbose >= logger.INFO:
                    homo, lumo = frontier[i]
                    gap = (lumo - homo) * nist.HARTREE2EV
                    comp.scf_summary['gap'] = gap
                    if homo+1e-3 > lumo:
                        logger.warn(comp, 'CNEO NUC HOMO %.15g == LUMO %.15g', homo, lumo)
                    else:
                        logger.info(comp, '  CNEO NUC HOMO = %.15g  LUMO = %.15g  gap/eV = %.5f',
                                    homo, lumo, gap)
        for i, t in enumerate(keys):
            mo_occ[t] = occ[i]
    return mo_occ


def _grouped_rdm1(components, mo_coeff, mo_occ):
    dm = {}
    groups = {}
    # Keep separate component-local AO matrices; only equal-sized matrices are
    # stacked along a temporary batch axis.
    for t in components:
        key = (mo_coeff[t].shape, mo_coeff[t].dtype)
        groups.setdefault(key, []).append(t)
    for keys in groups.values():
        coeff = cupy.stack([mo_coeff[t] for t in keys])
        occ = cupy.stack([mo_occ[t] for t in keys])
        occupied_index = cupy.argmax(occ, axis=1)
        gather_index = cupy.broadcast_to(occupied_index[:,None,None],
                                         (len(keys), coeff.shape[1], 1))
        occupied_coeff = cupy.take_along_axis(coeff, gather_index, axis=2)
        occupied_value = cupy.max(occ, axis=1)
        # Nuclear get_occ selects one orbital.  This is the original
        # C_occ n C_occ^H density formula evaluated for all components.
        dm_batch = cupy.matmul(occupied_coeff * occupied_value[:,None,None],
                               occupied_coeff.conj().swapaxes(-1, -2))
        for i, t in enumerate(keys):
            dm[t] = tag_array(dm_batch[i], occ_coeff=occupied_coeff[i],
                              mo_occ=mo_occ[t], mo_coeff=mo_coeff[t])
    return dm


def _grouped_grad(components, mo_coeff, mo_occ, fock):
    grad = {}
    groups = {}
    # Equal orbital dimensions give each component the same occupied-virtual
    # gradient layout, so the original contraction can use a batch axis.
    for t in components:
        key = (mo_coeff[t].shape, mo_coeff[t].dtype)
        groups.setdefault(key, []).append(t)
    for keys in groups.values():
        coeff = cupy.stack([mo_coeff[t] for t in keys])
        occ = cupy.stack([mo_occ[t] for t in keys])
        fock_batch = cupy.stack([fock[t] for t in keys])
        occidx = occ > 0
        viridx = ~occidx
        # This is scf.hf.get_grad's 2*(C_vir^H F C_occ).ravel() expression.
        coeff_t = coeff.swapaxes(1, 2)
        orbo = coeff_t[occidx].reshape(len(keys), -1, coeff.shape[1]).swapaxes(1, 2)
        orbv = coeff_t[viridx].reshape(len(keys), -1, coeff.shape[1]).swapaxes(1, 2)
        grad_batch = cupy.matmul(orbv.conj().swapaxes(1, 2),
                                 cupy.matmul(fock_batch, orbo)) * 2
        grad_batch = grad_batch.reshape(len(keys), -1)
        for i, t in enumerate(keys):
            grad[t] = grad_batch[i]
    return grad


def general_scf(method, charge=1, mass=1, is_nucleus=False, nuc_occ_state=0):
    '''Modify SCF (HF and DFT) method to support for general charge
    and general mass, such that positrons and nuclei can be calculated.

    Args:
        charge : float
            Charge of the particle. 1 means electron, -1 means positron.
        mass : float
            Mass of the particle in a.u. Nuclei will have high mass
        is_nucleus : bool
            If the particle is nucleus. Nucleus won't see PP and is
            considered a distinguishable single particle
        nuc_occ_state : int
            Select the nuclear orbital that is occupied. For Delta-SCF.
    '''
    assert isinstance(method, scf_gpu.hf.SCF)
    if isinstance(method, Component):
        method.charge = charge
        method.mass = mass
        method.is_nucleus = is_nucleus
        method.nuc_occ_state = nuc_occ_state
        method._vint = None
        return method
    return pyscf_lib.set_class(ComponentSCF(method, charge, mass, is_nucleus, nuc_occ_state),
                               (ComponentSCF, method.__class__))


class Component:
    __name_mixin__ = 'Component'


class ComponentSCF(Component):
    _keys = {'charge', 'mass', 'is_nucleus', 'nuc_occ_state'}

    def __init__(self, method, charge=1, mass=1, is_nucleus=False, nuc_occ_state=0):
        self.__dict__.update(method.__dict__)
        self.charge = charge
        self.mass = mass
        self.is_nucleus = is_nucleus
        self.nuc_occ_state = nuc_occ_state
        self._vint = None

    def undo_component(self):
        obj = pyscf_lib.view(self, pyscf_lib.drop_class(self.__class__, Component))
        del obj.charge, obj.mass, obj.is_nucleus, obj.nuc_occ_state, obj._vint
        return obj

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        from gpu4pyscf.pbc.gto.int1e import int1e_kin
        if mol._pseudo and not self.is_nucleus:
            from pyscf.gto import pp_int
            h = asarray(pp_int.get_gth_pp(mol)) * self.charge
        else:
            assert not mol.nucmod
            from gpu4pyscf.df.j_engine_3c2e import contract_int3c2e_auxvec
            nucmol = gto.mole.fakemol_for_charges(mol.atom_coords())
            Z = cupy.asarray(mol.atom_charges(), dtype=numpy.float64)
            h = contract_int3c2e_auxvec(mol, nucmol, -Z)
            # NEO scales the potential by the component charge and T by mass.
            h *= self.charge
        h += int1e_kin(mol) / self.mass

        if len(mol._ecpbas) > 0 and not self.is_nucleus:
            from gpu4pyscf.gto.ecp import get_ecp
            h += get_ecp(mol) * self.charge

        mm_mol = None
        if hasattr(mol, 'super_mol'):
            mm_mol = mol.super_mol.mm_mol
        elif hasattr(mol, 'mm_mol'):
            mm_mol = mol.mm_mol
        if mm_mol is not None:
            # Match GPU4PySCF qmmm.itrf: int1e_grids supports both point MM
            # charges and Gaussian MM charges through charge_exponents.
            from gpu4pyscf.gto.int3c1e import int1e_grids
            h -= _mm_charge_integrals(mm_mol, mol, int1e_grids) * self.charge
        return h

    def get_veff(self, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
        if mol is None: mol = self.mol
        with_ecoul = False
        if self.is_nucleus: # Nucleus does not have self-type interaction
            veff = cupy.zeros((mol.nao, mol.nao))
            if isinstance(self, scf_gpu.hf.KohnShamDFT):
                veff = tag_array(veff, ecoul=None, exc=0, vj=veff.copy())
            else:
                assert isinstance(self, scf_gpu.hf.RHF)
                with_ecoul = isinstance(dm, cupy.ndarray) and dm.ndim == 2
        else:
            if abs(self.charge) != 1.:
                raise NotImplementedError('General charge J/K with tag_array')
            if not isinstance(self, scf_gpu.hf.KohnShamDFT) and mol.nelectron == 1:
                # CPU HF1e is converted to UHF for GPU NEO. Skip its electronic
                # self J/K while retaining the inter-component potential below.
                if dm is None: dm = self.make_rdm1()
                veff = cupy.zeros_like(cupy.asarray(dm))
                # The omitted self J and K cancel; inter-component Coulomb is
                # added below and remains available for energy decomposition.
                veff = tag_array(veff, ecoul=0)
            elif hasattr(vhf_last, 'vhf_self'):
                veff = super().get_veff(mol, dm, dm_last, vhf_last.vhf_self, hermi)
            else:
                veff = super().get_veff(mol, dm, dm_last, vhf_last, hermi)
            with_ecoul = hasattr(veff, 'ecoul')

        if self._vint is None:
            if hasattr(mol, 'super_mol'):
                raise RuntimeError('ComponentSCF.get_veff cannot build the '
                                   'multicomponent effective potential without the inter-component '
                                   'cache. Call the parent NEO get_veff first, or pass a complete '
                                   'multicomponent vhf from the parent object.')
        else:
            # Save the self-type potential before adding inter-component terms.
            # Native SCF incremental J/K should see this object as vhf_last,
            # not the full NEO potential with _vint included.
            vhf_self = asarray(veff).copy()
            if hasattr(veff, '__dict__'):
                vhf_self = tag_array(vhf_self, **veff.__dict__)
                veff = tag_array(veff + self._vint, **veff.__dict__,
                                 vhf_self=vhf_self, vint=self._vint)
            else:
                veff = tag_array(veff + self._vint, vhf_self=vhf_self, vint=self._vint)
            if not isinstance(self, scf_gpu.hf.KohnShamDFT) and with_ecoul:
                dm_tot = cupy.asarray(dm)
                if isinstance(self, scf_gpu.uhf.UHF) and dm_tot.ndim == 3:
                    dm_tot = dm_tot[0] + dm_tot[1]
                ecoul = cupy.einsum('ij,ji->', dm_tot, self._vint).real.item() * .5
                if not self.is_nucleus:
                    ecoul += vhf_self.ecoul
                veff = tag_array(veff, ecoul=ecoul)
        return veff

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
        if self.is_nucleus:
            if self.mol.symmetry:
                raise NotImplementedError('Point-group symmetry for nuclear orbitals '
                                          'is not implemented')
            mo_energy = cupy.asarray(mo_energy)
            mo_occ = cupy.zeros_like(mo_energy)
            e_idx = cupy.argsort(mo_energy)
            nmo = mo_energy.size
            nocc = 1
            if self.verbose >= logger.INFO and nocc < nmo:
                homo, lumo = mo_energy[e_idx[nocc-1:nocc+1]].get()
                gap = (lumo - homo) * nist.HARTREE2EV
                self.scf_summary['gap'] = gap
                if homo+1e-3 > lumo:
                    logger.warn(self, 'CNEO NUC HOMO %.15g == LUMO %.15g', homo, lumo)
                else:
                    logger.info(self, '  CNEO NUC HOMO = %.15g  LUMO = %.15g  gap/eV = %.5f', homo, lumo, gap)
            elif nocc > nmo:
                raise RuntimeError(f'Failed to assign mo_occ. Nocc ({nocc}) > Nmo ({nmo})')
            mo_occ[e_idx[self.nuc_occ_state]] = self.mol.nnuc
            return mo_occ

        # Electronic occupations use the parent GPU implementation unless
        # fractional occupation was requested.
        if self.mol.symmetry:
            raise NotImplementedError('Point-group symmetry for electronic orbitals '
                                      'is not implemented')
        if self.mol.nhomo is None:
            return super().get_occ(mo_energy, mo_coeff)
        raise NotImplementedError('Fractional electronic occupation is not implemented')

    def get_init_guess(self, mol=None, key='minao', **kwargs):
        if self.is_nucleus:
            return 0

        # Build the electronic guess with the total molecular charge, then
        # restore the component charge used by NEO interactions.
        if mol is None: mol = self.mol
        charge = self.charge
        self.charge = abs(charge)
        dm = super().get_init_guess(mol.super_mol, key, **kwargs)
        self.charge = charge
        return dm

    def scf(self, dm0=None, **kwargs):
        raise AttributeError('scf should not be called from ComponentSCF')

    def dip_moment(self, mol=None, dm=None, unit='Debye', origin=None,
                   verbose=logger.NOTE, **kwargs):
        if self.is_nucleus:
            dm_cpu = _to_cpu(dm)
            return hf_cpu.ComponentSCF.dip_moment(self, mol, dm_cpu, unit, origin,
                                                  verbose, **kwargs)

        # Evaluate the electronic dipole with the total molecular charge.
        charge = self.mol.charge
        self.mol.charge = self.mol.super_mol.charge
        dip = super().dip_moment(mol, dm, unit, origin=origin, verbose=verbose, **kwargs)
        self.mol.charge = charge
        return dip

    def to_cpu(self):
        obj = self.undo_component().to_cpu()
        obj = hf_cpu.general_scf(obj, self.charge, self.mass, self.is_nucleus, self.nuc_occ_state)
        return utils.to_cpu(self, obj)

    mulliken_pop = NotImplemented
    mulliken_meta = NotImplemented


class InteractionCoulomb(hf_cpu.InteractionCoulomb):
    '''Inter-component Coulomb interactions.'''

    def __init__(self, mf1_type, mf1, mf2_type, mf2, max_memory, direct_scf_tol):
        super().__init__(mf1_type, mf1, mf2_type, mf2, max_memory, direct_scf_tol)
        self.mf1_unrestricted = isinstance(self.mf1, scf_gpu.uhf.UHF)
        self.mf2_unrestricted = isinstance(self.mf2, scf_gpu.uhf.UHF)


def _tag_vint_full_delta(vint_full, vint_delta, components):
    out = {}
    for t in components:
        out[t] = tag_array(vint_full[t] + vint_delta[t], vint_inc=vint_delta[t])
    return out

def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1,
             diis=None, diis_start_cycle=None, level_shift_factor=None,
             damp_factor=None, fock_last=None, diis_pos='both', diis_type=4,
             constraint_update=True):
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    h1e = {t: cupy.asarray(h1e[t]) for t in h1e}
    vhf = {t: cupy.asarray(vhf[t]) for t in vhf}
    f = {}
    for t, comp in mf.components.items():
        f[t] = h1e[t] + vhf[t]
        if not t.startswith('n') and isinstance(comp, scf_gpu.uhf.UHF) and f[t].ndim == 2:
            f[t] = cupy.asarray((f[t],) * 2)

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle

    from gpu4pyscf.neo import cdft
    is_cdft = isinstance(mf, cdft.CDFT)
    f0 = None
    position_error = None
    # CNEO constraint term
    # NOTE: even if not using DIIS, we still optimize f.
    if is_cdft:
        if diis_pos == 'pre' or diis_pos == 'both' or (cycle < 0 and diis is None):
            if constraint_update:
                # optimize the Lagrange multiplier in CNEO
                position_error = cdft.update_lagrange_multipliers(
                    mf, f, s1e, one_step=diis_type == 4 and cycle >= 0)

        # For DIIS type 1, preserve original matrices
        if diis_type == 1 and diis is not None and cycle >= diis_start_cycle:
            f0 = f.copy()

        fock_add = mf.get_fock_add_cdft()
        for t in fock_add:
            f[t] += fock_add[t]

    if cycle < 0 and diis is None:
        return f

    if s1e is None: s1e = mf.get_ovlp()
    if dm is None: dm = mf.make_rdm1()
    s1e = {t: cupy.asarray(s1e[t]) for t in s1e}
    dm = {t: cupy.asarray(dm[t]) for t in dm}
    for t, comp in mf.components.items():
        if not t.startswith('n') and isinstance(comp, scf_gpu.uhf.UHF) \
                and isinstance(dm[t], cupy.ndarray) and dm[t].ndim == 2:
            dm[t] = cupy.asarray((dm[t]*0.5,) * 2)

    if damp_factor is None:
        damp_factor = mf.damp
    if damp_factor is not None and 0 <= cycle < diis_start_cycle-1 and fock_last is not None \
            and abs(damp_factor) > 1e-12:
        raise NotImplementedError('Damping for multi-component SCF is not yet implemented.')

    if diis is not None and cycle >= diis_start_cycle:
        if diis.incore is None:
            # Follow scf.diis.CDIIS.update's history-memory estimate for the
            # packed component Focks and type-4 Lagrange multipliers.
            mem_avail = get_avail_mem()
            vector_bytes = sum(f_t.nbytes for f_t in f.values())
            if is_cdft and diis_type == 4:
                vector_bytes += 3 * sum(t.startswith('n') for t in f) * mf.f.itemsize
            diis.incore = vector_bytes*2 * (20+diis.space) < mem_avail
            if not diis.incore:
                logger.debug(mf, 'Large system detected. DIIS intermediates '
                             'are saved in the host memory')
        if is_cdft:
            keys = sorted(f.keys())
            shapes = {k: f[k].shape for k in keys}
            if getattr(diis, 'damp', 0):
                raise NotImplementedError('DIIS damping for CDFT is not implemented.')
            if diis_type != 1:
                variables = [f[k].ravel() for k in keys]
                if diis_type == 4:
                    nuclear_keys = sorted(t for t in mf.components if t.startswith('n'))
                    atom_indices = [mf.components[t].mol.atom_index for t in nuclear_keys]
                    variables.append(mf.f[atom_indices].ravel())
                f_flat = cupy.concatenate(variables)

            if diis_type == 1:
                f0_flat = cupy.concatenate([f0[k].ravel() for k in keys])
                # Type-1 CDFT extrapolates f0_flat while building the error
                # vector from the constrained Fock.  Bypass CDIIS.update for
                # this custom packed target/error-vector pair.
                errvec = diis._sdf_err_vec(s1e, dm, f)
                f_flat = lib.diis.DIIS.update(diis, f0_flat, xerr=errvec)
            elif diis_type == 2:
                f_flat = lib.diis.DIIS.update(diis, f_flat)
            elif diis_type == 3:
                # Equivalent to packing f and calling
                # lib.diis.DIIS.update(diis, f_flat, xerr=diis._sdf_err_vec(s1e, dm, f)).
                f = diis.update(s1e, dm, f)
                f_flat = None
            elif diis_type == 4:
                fock_error = diis._sdf_err_vec(s1e, dm, f)
                if position_error is None:
                    position_error = cdft.get_position_error(mf, f, s1e)
                error = cupy.concatenate((fock_error, position_error))
                f_flat = lib.diis.DIIS.update(diis, f_flat, xerr=error)
            else:
                logger.warn(mf, 'Unknown CDFT DIIS type %s; DIIS is disabled', diis_type)
                f_flat = None

            if f_flat is not None:
                # The type 1/2 CDFT paths bypass CDIIS.update, so reproduce
                # CDIIS' post-update rollback trimming after the raw DIIS call.
                if diis.rollback > 0 and len(diis._bookkeep) == diis.space:
                    diis._bookkeep = diis._bookkeep[-diis.rollback:]
                # Reconstruct dictionary
                offset = 0
                f_new = {}
                for k in keys:
                    size = f[k].size
                    f_new[k] = f_flat[offset:offset+size].reshape(shapes[k])
                    offset += size
                f = f_new

                if diis_type == 4:
                    size = len(nuclear_keys) * 3
                    mf.f[atom_indices] = f_flat[offset:offset+size].reshape(-1,3)
                    offset += size
                    fock_add = mf.get_fock_add_cdft()

            if diis_type == 1:
                for t in fock_add:
                    f[t] += fock_add[t]
        else:
            f = diis.update(s1e, dm, f)

    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if level_shift_factor is not None and abs(level_shift_factor) > 1e-12:
        raise NotImplementedError('Level shift for multi-component SCF is not yet implemented.')

    # Post-DIIS CDFT optimization
    if (is_cdft and constraint_update and (diis_pos == 'post' or diis_pos == 'both')):
        f0 = {}
        for t in f:
            if t.startswith('n'):
                f0[t] = f[t] - fock_add[t]
            else:
                f0[t] = f[t]

        cdft.update_lagrange_multipliers(mf, f0, s1e, one_step=diis_type == 4)

        fock_add = mf.get_fock_add_cdft()
        for t in fock_add:
            f[t] = f0[t] + fock_add[t]
    return f


def _kernel(mf, conv_tol=1e-10, conv_tol_grad=None,
            dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    conv_tol = mf.conv_tol
    mol = mf.mol
    verbose = mf.verbose
    log = logger.new_logger(mf, verbose)
    t0 = t1 = log.init_timer()
    if conv_tol_grad is None:
        conv_tol_grad = conv_tol**.5
        log.info('Set gradient conv threshold to %g', conv_tol_grad)

    if dm0 is None:
        dm0 = mf.get_init_guess(mol, mf.init_guess, **kwargs)
        t1 = log.timer_debug1('generating initial guess', *t1)
    else:
        dm0 = mf.get_init_guess(mol, dm0, **kwargs)

    e_dm = dm0.get('e') if isinstance(dm0, dict) else None
    mo_coeff0 = mo_occ0 = None
    if hasattr(e_dm, 'mo_coeff') and hasattr(e_dm, 'mo_occ'):
        mo_coeff0 = cupy.asarray(e_dm.mo_coeff)
        mo_occ0 = cupy.asarray(e_dm.mo_occ)
    dm0 = {t: cupy.asarray(dm0[t], order='C') for t in dm0}
    if mo_coeff0 is not None and mo_occ0 is not None:
        dm0['e'] = tag_array(dm0['e'], mo_coeff=mo_coeff0, mo_occ=mo_occ0)

    dm, dm0 = dm0, None
    vhf = mf.get_veff(mol, dm)

    h1e = {t: cupy.asarray(v) for t, v in mf.get_hcore(mol).items()}
    s1e = {t: cupy.asarray(v) for t, v in mf.get_ovlp(mol).items()}
    e_tot = mf.energy_tot(dm, h1e, vhf)
    log.info('init E= %.15g', e_tot)
    x_orth = mf.check_linear_dependency(s1e, log)
    t1 = log.timer('SCF initialization', *t0)
    scf_conv = False

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        mo_energy, mo_coeff = mf.eig(fock, s1e, x=x_orth)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback
        mf_diis.damp = mf.diis_damp
        mf_diis.Corth = _to_gpu(x_orth)
    else:
        mf_diis = None

    dump_chk = dump_chk and mf.chkfile is not None
    if dump_chk:
        chkfile.save_mol(mol, mf.chkfile)

    fock_last = None
    mf.cycles = 0
    for cycle in range(mf.max_cycle):
        t0 = log.init_timer()
        mo_coeff = mo_occ = mo_energy = fock = None
        dm_last = dm
        last_hf_e = e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, mf_diis, fock_last=fock_last)
        t1 = log.timer_debug1('DIIS', *t0)
        mo_energy, mo_coeff = mf.eig(fock, s1e, x=x_orth)
        if mf.damp is not None:
            fock_last = fock
        fock = None
        t1 = log.timer_debug1('eig', *t1)

        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        dm = {t: asarray(dm[t]) for t in dm} # Remove the attached attributes
        t1 = log.timer_debug1('veff', *t1)

        fock = mf.get_fock(h1e, s1e, vhf, dm,
                           constraint_update=False)  # = h1e + vhf, no DIIS
        e_tot = mf.energy_tot(dm, h1e, vhf)
        grad = mf.get_grad(mo_coeff, mo_occ, fock)
        norm_gorb = {t: cupy.linalg.norm(grad[t]) for t in grad}

        norm_ddm = {t: cupy.linalg.norm(dm[t]-dm_last[t]) for t in dm}
        t1 = log.timer(f'cycle={cycle+1}', *t0)

        log.info('cycle= %d E= %.15g  delta_E= %4.3g  |g_e|= %4.3g  |ddm_e|= %4.3g',
                 cycle+1, e_tot, e_tot-last_hf_e, norm_gorb['e'], norm_ddm['e'])
        for t in grad:
            if not t.startswith('e'):
                log.info(f'    |g_{t}|= %4.3g  |ddm_{t}|= %4.3g', norm_gorb[t], norm_ddm[t])

        if dump_chk:
            mf.dump_chk({
                'mol': mol,
                'mo_energy': mo_energy,
                'mo_occ': mo_occ,
                'mo_coeff': mo_coeff,
                'e_tot': e_tot,
            })

        if callable(callback):
            callback(locals())

        e_diff = abs(e_tot-last_hf_e)
        if (e_diff < conv_tol and norm_gorb['e'] < conv_tol_grad):
            scf_conv = True
            break
    else:
        log.warn("SCF failed to converge")

    mf.cycles = cycle + 1
    if scf_conv and mf.level_shift is not None:
        mo_coeff = mo_occ = mo_energy = mf_diis = None
        # An extra diagonalization, to remove level shift
        from gpu4pyscf.neo import cdft
        if isinstance(mf, cdft.CDFT):
            # Rebuild the Fock matrix to fully optimize the CNEO constraints
            fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf
        mo_energy, mo_coeff = mf.eig(fock, s1e, x=x_orth)
        fock = None
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
        grad = mf.get_grad(mo_coeff, mo_occ, fock)
        norm_gorb = {t: cupy.linalg.norm(grad[t]) for t in grad}
        norm_ddm = {t: cupy.linalg.norm(dm[t]-dm_last[t]) for t in dm}

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if abs(e_tot-last_hf_e) < conv_tol or norm_gorb['e'] < conv_tol_grad:
            scf_conv = True
        else:
            log.warn("Level-shifted SCF extra cycle failed to converge")
            scf_conv = False
        log.info('Extra cycle  E= %.15g  delta_E= %4.3g  |g_e|= %4.3g  |ddm_e|= %4.3g',
                 e_tot, e_tot-last_hf_e, norm_gorb['e'], norm_ddm['e'])
        for t in grad:
            if not t.startswith('e'):
                log.info(f'    |g_{t}|= %4.3g  |ddm_{t}|= %4.3g', norm_gorb[t], norm_ddm[t])
        if dump_chk:
            mf.dump_chk(locals())

    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


def scf(mf, dm0=None, **kwargs):
    cput0 = logger.init_timer(mf)

    mf.dump_flags()
    mf.build(mf.mol)

    if dm0 is None and mf.mo_coeff is not None and mf.mo_occ is not None:
        # Initial guess from existing wavefunction
        dm0 = mf.make_rdm1()

    if mf.max_cycle > 0 or mf.mo_coeff is None:
        mf.converged, mf.e_tot, \
                mf.mo_energy, mf.mo_coeff, mf.mo_occ = \
                _kernel(mf, mf.conv_tol, mf.conv_tol_grad,
                        dm0=dm0, callback=mf.callback,
                        conv_check=mf.conv_check, **kwargs)
        for t, comp in mf.components.items():
            comp.mo_energy = mf.mo_energy[t]
            comp.mo_coeff = mf.mo_coeff[t]
            comp.mo_occ = mf.mo_occ[t]
            comp.converged = mf.converged
    else:
        # Avoid updating SCF orbitals in non-SCF initialization.
        mf.e_tot = _kernel(mf, mf.conv_tol, mf.conv_tol_grad,
                           dm0=dm0, callback=mf.callback,
                           conv_check=mf.conv_check, **kwargs)[1]

    logger.timer(mf, 'Multicomponent-SCF', *cput0)
    mf._finalize()
    return mf.e_tot


def energy_tot(mf, dm=None, h1e=None, vhf=None):
    nuc = mf.energy_nuc()
    mf.scf_summary['nuc'] = nuc.real

    e_tot = mf.energy_elec(dm, h1e, vhf)[0] + nuc
    if mf.disp is not None:
        mf.components['e'].disp = mf.disp
    if mf.components['e'].do_disp():
        if 'dispersion' in mf.components['e'].scf_summary:
            e_tot += mf.components['e'].scf_summary['dispersion']
        else:
            e_disp = mf.components['e'].get_dispersion()
            mf.components['e'].scf_summary['dispersion'] = e_disp
            e_tot += e_disp
        mf.scf_summary['dispersion'] = mf.components['e'].scf_summary['dispersion']

    if isinstance(e_tot, cupy.ndarray):
        e_tot = e_tot.get()
    return e_tot


def _grouped_energy(components, dm, h1e, vhf):
    e_elec = 0
    e2 = 0
    groups = {}
    # HF and KS effective potentials carry different energy metadata.  Group
    # only components with equal local AO layouts and the same method type.
    for t, comp in components.items():
        key = (dm[t].shape, dm[t].dtype, isinstance(comp, scf_gpu.hf.KohnShamDFT))
        groups.setdefault(key, []).append(t)
    for keys in groups.values():
        # The leading axis batches independent component-local matrices; no
        # cross-component AO blocks are formed.
        dm_batch = cupy.stack([dm[t] for t in keys])
        h1e_batch = cupy.stack([h1e[t] for t in keys])
        e1 = cupy.einsum('nij,nji->n', h1e_batch, dm_batch).real
        if isinstance(components[keys[0]], scf_gpu.hf.KohnShamDFT):
            # KS get_veff already records Coulomb and XC energies.
            e1 = e1.get()
            energies = [(e1[i], vhf[t].ecoul.real + vhf[t].exc.real) for i, t in enumerate(keys)]
        else:
            # Preserve scf.hf.energy_elec's E2 = Tr[V_hf D]/2 contraction.
            vhf_batch = cupy.stack([vhf[t] for t in keys])
            e2_batch = cupy.einsum('nij,nji->n', vhf_batch, dm_batch).real * .5
            energies = cupy.stack((e1, e2_batch), axis=1).get()
        for i, t in enumerate(keys):
            comp = components[t]
            e1_t, e2_t = energies[i]
            ecoul = vhf[t].ecoul.real
            comp.scf_summary['e1'] = e1_t
            comp.scf_summary['e2'] = e2_t
            comp.scf_summary['coul'] = ecoul
            if hasattr(vhf[t], 'exc'):
                comp.scf_summary['exc'] = vhf[t].exc.real
            else:
                comp.scf_summary['exc'] = e2_t - ecoul
            e_elec += e1_t + e2_t
            e2 += e2_t
    return e_elec, e2


def energy_elec(mf, dm=None, h1e=None, vhf=None):
    if dm is None: dm = mf.make_rdm1()
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    mf.scf_summary['e1'] = 0
    mf.scf_summary['e2'] = 0
    e_elec = 0
    e_coul = 0
    ecoul = 0
    with_ecoul = True
    nuclear_components = {t: comp for t, comp in mf.components.items()
                          if t.startswith('n')}
    for t, comp in mf.components.items():
        if t.startswith('n'):
            continue
        logger.debug(mf, f'Component: {t}')
        e_elec_t, e_coul_t = comp.energy_elec(dm[t], h1e[t], vhf[t])
        e_elec += e_elec_t
        e_coul += e_coul_t
        mf.scf_summary['e1'] += comp.scf_summary['e1']
        mf.scf_summary['e2'] += comp.scf_summary['e2']
        if hasattr(vhf[t], 'ecoul'):
            ecoul += vhf[t].ecoul.real
        else:
            with_ecoul = False
    if nuclear_components:
        e_elec_n, e_coul_n = _grouped_energy(nuclear_components, dm, h1e, vhf)
        e_elec += e_elec_n
        e_coul += e_coul_n
        for t in nuclear_components:
            comp = nuclear_components[t]
            mf.scf_summary['e1'] += comp.scf_summary['e1']
            mf.scf_summary['e2'] += comp.scf_summary['e2']
            if hasattr(vhf[t], 'ecoul'):
                ecoul += vhf[t].ecoul.real
            else:
                with_ecoul = False
    if with_ecoul:
        mf.scf_summary['coul'] = ecoul
        exx = mf.scf_summary['e2'] - ecoul
        mf.scf_summary['exc'] = exx
    return e_elec, e_coul


class HF(scf_gpu.hf.SCF):
    '''Multicomponent Hartree-Fock'''

    _keys = scf_gpu.hf.SCF._keys.union({'unrestricted', 'components', 'interactions',})

    def __init__(self, mol, unrestricted=False):
        super().__init__(mol)
        self.unrestricted = unrestricted
        self.components = {}
        for t, comp in self.mol.components.items():
            if t.startswith('n'):
                charge = -1. * self.mol.atom_charge(comp.atom_index)
                mass = self.mol.mass[comp.atom_index] * nist.ATOMIC_MASS / nist.E_MASS
                self.components[t] = general_scf(scf_gpu.RHF(comp), charge=charge, mass=mass,
                                                 is_nucleus=True, nuc_occ_state=0)
            else:
                if self.unrestricted:
                    mf = scf_gpu.UHF(comp)
                elif getattr(comp, 'nhomo', None) is not None or comp.spin != 0:
                    mf = scf_gpu.UHF(comp)
                else:
                    mf = scf_gpu.RHF(comp)
                charge = -1. if t.startswith('p') else 1.
                self.components[t] = general_scf(mf, charge=charge)
        self.interactions = hf_cpu.generate_interactions(self.components, InteractionCoulomb,
                                                         self.max_memory, self.direct_scf_tol)

    get_fock = get_fock

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        if self.mol.mm_mol is not None:
            logger.info(self, '** Add background charges for %s **',
                        self.__class__.__name__)
        return self

    def check_linear_dependency(self, s, verbose=None):
        x = {}
        nuclear_representatives = {}
        for t, comp in self.components.items():
            if t.startswith('n'):
                representative = nuclear_representatives.setdefault(s[t].data.ptr, t)
                if representative != t:
                    x[t] = x[representative]
                    continue
            x[t] = comp.check_linear_dependency(s[t], verbose)
        return x
    check_sanity = hf_cpu.HF.check_sanity
    build = hf_cpu.HF.build

    def eig(self, h, s, overwrite=False, x=None):
        energy = {}
        coeff = {}
        nuclear_components = {t: comp for t, comp in self.components.items()
                              if t.startswith('n')}
        # Preserve each nonnuclear component's own eig implementation.
        for t, comp in self.components.items():
            if not t.startswith('n'):
                x_t = None if x is None else x[t]
                energy[t], coeff[t] = comp.eig(h[t], s[t], overwrite=overwrite, x=x_t)
        if nuclear_components:
            # Nuclear matrices are independent but often equal-sized, allowing
            # the same eig operations to share a leading batch dimension.
            energy_n, coeff_n = _grouped_eig(nuclear_components, h, s, overwrite=overwrite, x=x)
            energy.update(energy_n)
            coeff.update(coeff_n)
        return energy, coeff

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        hcore = {}
        nuclear_mols = {t: comp for t, comp in mol.components.items() if t.startswith('n')}
        for t, comp in mol.components.items():
            if not t.startswith('n'):
                hcore[t] = self.components[t].get_hcore(mol=comp)
        if nuclear_mols:
            nuclear_components = {t: self.components[t] for t in nuclear_mols}
            hcore.update(_grouped_hcore(nuclear_components, nuclear_mols))
        return hcore

    def get_ovlp(self, mol=None):
        from gpu4pyscf.neo import int1e
        if mol is None: mol = self.mol
        ovlp = {}
        nuclear_mols = {t: comp for t, comp in mol.components.items() if t.startswith('n')}
        for t, comp in mol.components.items():
            if not t.startswith('n'):
                ovlp[t] = self.components[t].get_ovlp(mol=comp)
        if nuclear_mols:
            ovlp.update(int1e.Int1eOpt(nuclear_mols).get_ovlp())
        return ovlp

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
        mo_occ = {}
        nuclear_components = {t: comp for t, comp in self.components.items()
                              if t.startswith('n')}
        for t, comp in self.components.items():
            if not t.startswith('n'):
                coeff = mo_coeff.get(t) if mo_coeff is not None and \
                        isinstance(mo_coeff, dict) else None
                mo_occ[t] = comp.get_occ(mo_energy[t], coeff)
        # Nuclear occupations use nuc_occ_state rather than the electronic
        # Aufbau occupation implemented by the component SCF class.
        mo_occ.update(_grouped_occ(nuclear_components, mo_energy))
        if 'gap' in self.components['e'].scf_summary:
            self.scf_summary['gap'] = self.components['e'].scf_summary['gap']
        return mo_occ

    def get_grad(self, mo_coeff, mo_occ, fock):
        grad = {}
        nuclear_components = {t: comp for t, comp in self.components.items()
                              if t.startswith('n')}
        for t, comp in self.components.items():
            if not t.startswith('n'):
                grad[t] = comp.get_grad(mo_coeff[t], mo_occ[t], fock[t])
        # Nuclear occupied-virtual contractions share a batch dimension when
        # their orbital dimensions agree.
        grad.update(_grouped_grad(nuclear_components, mo_coeff, mo_occ, fock))
        return grad

    def get_init_guess(self, mol=None, key='minao', **kwargs):
        from gpu4pyscf.neo import int1e
        dm_guess = {}
        if not isinstance(key, str):
            if isinstance(key, dict): # several components are given
                dm_guess = key
            else: # numpy.ndarray
                dm_guess['e'] = key   # only e_guess is provided
            key = 'minao' # for remaining components, use default minao guess
        if mol is None: mol = self.mol
        if 'e' not in dm_guess:
            dm_guess['e'] = self.components['e'].get_init_guess(mol.components['e'], key, **kwargs)

        if 'p' in self.components and 'p' not in dm_guess:
            dm_guess['p'] = self.components['p'].get_init_guess(mol.components['p'], key, **kwargs)

        nuc_types = tuple(t for t in self.components if t.startswith('n') and t not in dm_guess)
        vint = self._get_init_guess_vint(nuc_types, dm_guess) if nuc_types else {}
        nuc_mols = {}
        for t in nuc_types:
            comp = self.components[t]
            mol_tmp = neo.Mole()
            # Do not invoke possibly expensive QMMM during init guess
            mol_tmp.build(quantum_nuc=[comp.mol.atom_index],
                          nuc_basis=mol.nuclear_basis,
                          mm_mol=None, dump_input=False, parse_arg=False,
                          verbose=mol.verbose, output=mol.output,
                          max_memory=mol.max_memory, atom=mol.atom, unit=mol.unit,
                          nucmod=mol.nucmod, ecp=mol.ecp, pseudo=mol.pseudo,
                          charge=mol.charge, spin=mol.spin, symmetry=mol.symmetry,
                          symmetry_subgroup=mol.symmetry_subgroup, cart=mol.cart,
                          magmom=mol.magmom)
            nuc_mols[t] = mol_tmp.components[t]

        if nuc_mols:
            nuc_components = {t: self.components[t] for t in nuc_types}
            int1e_opt = int1e.Int1eOpt(nuc_mols)
            hcore = _grouped_hcore(nuc_components, nuc_mols, int1e_opt)
            ovlp = int1e_opt.get_ovlp()
            fock = {t: hcore[t] + vint[t] for t in nuc_types}
            mo_energy, mo_coeff = _grouped_eig(nuc_components, fock, ovlp)
            mo_occ = _grouped_occ(nuc_components, mo_energy)
            dm_guess.update(_grouped_rdm1(nuc_components, mo_coeff, mo_occ))
        return dm_guess

    def _get_init_guess_vint(self, output_components, dm_guess):
        dm_guess = _to_cpu(dm_guess)
        vint = hf_cpu.HF._get_init_guess_vint(self, output_components, dm_guess)
        return {t: cupy.asarray(v) for t, v in vint.items()}

    def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        dm = {}
        nuclear_components = {t: comp for t, comp in self.components.items()
                              if t.startswith('n')}
        for t, comp in self.components.items():
            if not t.startswith('n'):
                coeff = comp.mo_coeff if mo_coeff is None else mo_coeff[t]
                occ = comp.mo_occ if mo_occ is None else mo_occ[t]
                dm[t] = comp.make_rdm1(coeff, occ, **kwargs)
        # Keep nuclear density matrices component-local while batching equal
        # dimensions in the contraction.
        dm.update(_grouped_rdm1(nuclear_components, mo_coeff, mo_occ))
        return dm

    energy_elec = energy_elec
    energy_tot = energy_tot
    kernel = scf = scf
    as_scanner = hf_cpu.as_scanner
    mulliken_meta = pop = NotImplemented
    mulliken_pop = NotImplemented
    canonicalize = NotImplemented

    def energy_nuc(self):
        nuc = self.components['e'].energy_nuc()
        if self.mol.mm_mol is None:
            return nuc

        # Cache the classical QM-MM term as in qmmm.itrf.QMMMSCF.
        # NEO excludes quantum nuclei through the electronic component charges
        # and retains the Gaussian MM model supported by CPU NEO.
        e_nuc_ext = getattr(self, '_e_nuc_ext', None)
        if e_nuc_ext is None:
            mm_mol = self.mol.mm_mol
            coords = mm_mol.atom_coords()
            charges = mm_mol.atom_charges()
            if mm_mol.charge_model == 'gaussian':
                expnts = numpy.sqrt(mm_mol.get_zetas())
            mol_e = self.components['e'].mol
            e_nuc_ext = 0.
            for j in range(mol_e.natm):
                q2, r2 = mol_e.atom_charge(j), mol_e.atom_coord(j)
                r = pyscf_lib.norm(r2-coords, axis=1)
                if mm_mol.charge_model != 'gaussian':
                    e_nuc_ext += q2*(charges/r).sum()
                else:
                    e_nuc_ext += q2*(charges*erf(expnts*r)/r).sum()
            self._e_nuc_ext = e_nuc_ext
        nuc += e_nuc_ext
        return nuc

    def dump_chk(self, envs):
        if self.chkfile:
            assert isinstance(envs, dict)
            chkfile.dump_scf(self.mol, self.chkfile, envs['e_tot'],
                             _to_cpu(envs['mo_energy']),
                             _to_cpu(envs['mo_coeff']),
                             _to_cpu(envs['mo_occ']), overwrite_mol=False)

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        raise AttributeError('get_jk should not be called from multi-component SCF')

    def get_j(self, mol=None, dm=None, hermi=1, omega=None):
        raise AttributeError('get_j should not be called from multi-component SCF')

    def get_k(self, mol=None, dm=None, hermi=1, omega=None):
        raise AttributeError('get_k should not be called from multi-component SCF')

    def nuc_grad_method(self):
        return self.Gradients()

    def Gradients(self):
        from gpu4pyscf.neo import grad
        return grad.Gradients(self)

    def get_veff(self, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        vint = self._get_vint(mol, dm, dm_last, vhf_last)
        vhf = {}
        for t, comp in self.components.items():
            dm_last_t = dm_last[t] if isinstance(dm_last, dict) else None
            vhf_last_t = vhf_last[t] if isinstance(vhf_last, dict) else None
            vint_coul = vint[t].vj if hasattr(vint[t], 'vj') else vint[t]
            vint_exc = vint[t].exc if hasattr(vint[t], 'exc') else 0
            vint_inc = getattr(vint[t], 'vint_inc', 0)
            comp._vint = cupy.asarray(vint[t])
            vhf[t] = comp.get_veff(mol.components[t], dm[t], dm_last_t,
                                   vhf_last_t, hermi)
            if isinstance(comp, scf_gpu.hf.KohnShamDFT):
                vhf_self = vhf[t].vhf_self
                # Include the intercomponent EPC contribution in the
                # exchange-correlation energy tag.
                exc = vhf_self.exc + vint_exc
                ecoul = vhf_self.ecoul
                # Add the intercomponent Coulomb energy for ground-state DMs.
                dm_t = dm[t]
                if comp.is_nucleus:
                    ground_state = (isinstance(dm_t, cupy.ndarray) and dm_t.ndim == 2)
                else:
                    ground_state = ecoul is not None
                if ground_state:
                    if isinstance(comp, scf_gpu.uhf.UHF):
                        if not isinstance(dm_t, cupy.ndarray):
                            dm_t = asarray(dm_t)
                        if dm_t.ndim == 2:  # RHF DM
                            dm_tot = dm_t
                        else:
                            dm_tot = dm_t[0] + dm_t[1]
                    else:
                        dm_tot = dm_t
                    ecoul_vint = cupy.einsum('ij,ji->', dm_tot, vint_coul).real.item() * .5
                    if ecoul is not None:
                        ecoul += ecoul_vint
                    elif comp.is_nucleus:
                        ecoul = ecoul_vint
                    else:
                        raise RuntimeError(
                            f'Missing self Coulomb energy tag for component {t}')
                tags = {'ecoul': ecoul, 'exc': exc, 'vhf_self': vhf_self,
                        'vint': comp._vint, 'vint_inc': vint_inc}
                if hasattr(vhf_self, 'vj'):
                    tags['vj'] = vhf_self.vj + vint_coul
                vhf[t] = tag_array(vhf[t], **tags)
            else:
                vhf[t] = tag_array(vhf[t], vint_inc=vint_inc)
        return vhf

    def _get_vint(self, mol=None, dm=None, dm_last=None, vhf_last=None, **kwargs):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        incremental_j = self.direct_scf and isinstance(dm_last, dict) \
            and isinstance(vhf_last, dict) and \
            all(t in vhf_last and hasattr(vhf_last[t], 'vint_inc') for t in self.components)
        vint_full, vint_delta = hf_cpu._init_vint_full_delta(self.components, vhf_last,
                                                             incremental_j)
        dm_cpu = None
        ddm_cpu = None
        vint_full_cpu = None
        vint_delta_cpu = None
        for t_pair, interaction in self.interactions.items():
            incremental_vint = interaction._is_direct_vint()
            if incremental_vint and incremental_j:
                if ddm_cpu is None:
                    ddm = {}
                    for t, dm_ in dm.items():
                        ddm[t] = cupy.asarray(dm_) - cupy.asarray(dm_last[t])
                    ddm_cpu = _to_cpu(ddm)
                dm_interaction = ddm_cpu
            else:
                if dm_cpu is None:
                    dm_cpu = _to_cpu(dm)
                dm_interaction = dm_cpu
            if incremental_vint:
                if vint_delta_cpu is None:
                    vint_delta_cpu = _to_cpu(vint_delta)
                    vint_delta = vint_delta_cpu
            else:
                if vint_full_cpu is None:
                    vint_full_cpu = _to_cpu(vint_full)
                    vint_full = vint_full_cpu
            v = interaction.get_vint(dm_interaction)
            hf_cpu._accumulate_vint(vint_full, vint_delta, v, t_pair, incremental_vint)
        for t in self.components:
            if isinstance(vint_full[t], numpy.ndarray):
                vint_full[t] = cupy.asarray(vint_full[t])
            if isinstance(vint_delta[t], numpy.ndarray):
                vint_delta[t] = cupy.asarray(vint_delta[t])
        return _tag_vint_full_delta(vint_full, vint_delta, self.components)

    def analyze(self, verbose=None, with_meta_lowdin=WITH_META_LOWDIN,
                **kwargs):
        return self.to_cpu().analyze(verbose=verbose,
                                     with_meta_lowdin=with_meta_lowdin,
                                     **kwargs)

    def dip_moment(self, mol=None, dm=None, unit='Debye', origin=None,
                   verbose=logger.NOTE, **kwargs):
        return self.to_cpu().dip_moment(mol, _to_cpu(dm), unit, origin, verbose, **kwargs)

    def density_fit(self, auxbasis=None, with_df=None, ee_only_dfj=False,
                    df_ne=True, df_nn=True, df_ne_scheme='global',
                    nuc_auxbasis=None, nuc_auxbasis_beta=2.0,
                    nuc_auxbasis_lmax=None,
                    df_ne_component_vint=False, df_ne_j_engine='direct'):
        from gpu4pyscf.neo import df
        return df.density_fit(self, auxbasis=auxbasis, with_df=with_df,
                              ee_only_dfj=ee_only_dfj, df_ne=df_ne,
                              df_nn=df_nn, df_ne_scheme=df_ne_scheme,
                              nuc_auxbasis=nuc_auxbasis,
                              nuc_auxbasis_beta=nuc_auxbasis_beta,
                              nuc_auxbasis_lmax=nuc_auxbasis_lmax,
                              df_ne_component_vint=df_ne_component_vint,
                              df_ne_j_engine=df_ne_j_engine)

    def _finalize(self):
        super()._finalize()
        if self.mol.symmetry or any(getattr(comp.mol, 'symmetry', False)
                                    for comp in self.components.values()):
            logger.warn(self, 'GPU NEO symmetry support is incomplete; '
                        '_finalize still uses vanilla GPU orbital ordering')
        return self

    def copy(self):
        new = super().copy()
        if hasattr(self, 'f') and self.f is not None:
            new.f = cupy.array(self.f, copy=True)
        new.components = {}
        for t, comp in self.components.items():
            new.components[t] = general_scf(comp.undo_component().copy(), charge=comp.charge,
                                            mass=comp.mass, is_nucleus=comp.is_nucleus,
                                            nuc_occ_state=comp.nuc_occ_state)
        new.interactions = hf_cpu.generate_interactions(new.components, InteractionCoulomb,
                                                        new.max_memory, new.direct_scf_tol)
        return new

    def reset(self, mol=None):
        '''Reset mol and relevant attributes associated to the old mol object'''
        self._e_nuc_ext = None
        if mol is not None:
            self.mol = mol
        super().reset(mol=mol)
        components = self.components.copy()
        if components.keys() != self.mol.components.keys():
            self.mo_coeff = None
        self.components.clear()
        for t, comp in self.mol.components.items():
            is_nucleus = t.startswith('n')
            unrestricted = not is_nucleus and (self.unrestricted or comp.spin != 0 or
                                               getattr(comp, 'nhomo', None) is not None)
            mf_class = scf_gpu.uhf.UHF if unrestricted else scf_gpu.hf.RHF
            mf = components.get(t)
            # Nuclear membership and electronic spin can change independently.
            # Keep compatible component objects, including their DF decoration.
            if isinstance(mf, mf_class):
                mf.reset(comp)
            else:
                mf = mf_class(comp)
                self.mo_coeff = None
            if is_nucleus:
                self.components[t] = general_scf(mf,
                                                 charge=-1. * self.mol.atom_charge(comp.atom_index),
                                                 mass=self.mol.mass[comp.atom_index] * nist.ATOMIC_MASS
                                                      / nist.E_MASS,
                                                 is_nucleus=True,
                                                 nuc_occ_state=getattr(mf, 'nuc_occ_state', 0))
            else:
                charge = -1. if t.startswith('p') else 1.
                self.components[t] = general_scf(mf, charge=charge)
        # Recreate pair molecules and spin flags from the reset components.
        self.interactions.clear()
        self.interactions.update(hf_cpu.generate_interactions(
            self.components, InteractionCoulomb,
            self.max_memory, self.direct_scf_tol))

        return self

    def to_cpu(self):
        obj = hf_cpu.HF(self.mol, unrestricted=self.unrestricted)
        for key in self._keys:
            if key in ('components', 'interactions'):
                continue
            if hasattr(self, key):
                setattr(obj, key, _to_cpu(getattr(self, key)))
        obj.components = {t: comp.to_cpu() for t, comp in self.components.items()}
        obj.interactions = hf_cpu.generate_interactions(obj.components, hf_cpu.InteractionCoulomb,
                                                        obj.max_memory, obj.direct_scf_tol)
        return obj


def from_cpu(mf):
    out = HF(mf.mol, unrestricted=mf.unrestricted)
    for key, val in mf.__dict__.items():
        if key in ('components', 'interactions'):
            continue
        setattr(out, key, _to_gpu(val))
    out.components = {t: comp.to_gpu() for t, comp in mf.components.items()}
    out.interactions = hf_cpu.generate_interactions(out.components, InteractionCoulomb,
                                                    out.max_memory, out.direct_scf_tol)
    return out
