import copy
import warnings
from concurrent.futures import ThreadPoolExecutor
import cupy
from pyscf import gto
from pyscf import scf as scf_cpu
from pyscf.data import nist
from pyscf.neo import hf as hf_cpu
from pyscf.neo import ks as ks_cpu
from gpu4pyscf import dft, scf
from gpu4pyscf.__config__ import num_devices
from gpu4pyscf.dft import numint
from gpu4pyscf.dft import rks
from gpu4pyscf.lib import utils
from gpu4pyscf.lib.cupy_helper import add_sparse, reduce_to_device, release_gpu_stack, tag_array
from gpu4pyscf.neo import hf


def precompute_epc_electron(epc, rho_e):
    params = {
        '17-1': (2.35, 2.4, 3.2),
        '17-2': (2.35, 2.4, 6.6),
        '18-1': (1.8, 0.1, 0.03),
        '18-2': (3.9, 0.5, 0.06)
    }
    if isinstance(epc, dict):
        epc_type = epc.get('epc_type', '17-2')
        if epc_type in ('17', '18'):
            a = epc['a']
            b = epc['b']
            c = epc['c']
        else:
            if epc_type not in params:
                raise ValueError(f'Unknown EPC type: {epc_type}')
            a, b, c = params[epc_type]
    else:
        epc_type = epc
        if epc_type not in params:
            raise ValueError(f'Unknown EPC type: {epc_type}')
        a, b, c = params[epc_type]

    common = {'a': a, 'b': b, 'c': c, 'type': epc_type}
    if epc_type.startswith('17'):
        common['rho_e'] = rho_e
    else:
        common['rho_e'] = rho_e
        common['rho_e_cbrt'] = cupy.cbrt(rho_e)
        common['rho_e_cbrt4'] = common['rho_e_cbrt']**4
    return common


def eval_epc(common, rho_n):
    epc_type = common['type']
    a = common['a']
    b = common['b']
    c = common['c']
    rho_e = common['rho_e']

    if epc_type.startswith('17'):
        rho_prod = cupy.multiply(rho_e, rho_n)
        rho_sqrt = cupy.sqrt(rho_prod)
        denom = a - b * rho_sqrt + c * rho_prod
        denom2 = cupy.square(denom)

        exc = -rho_e / denom
        numer_n = -a * rho_e + 0.5 * b * rho_e * rho_sqrt
        vxc_n = numer_n / denom2
        numer_e = -a * rho_n + 0.5 * b * rho_n * rho_sqrt
        vxc_e = numer_e / denom2
    else:
        rho_e_cbrt = common['rho_e_cbrt']
        rho_e_cbrt4 = common['rho_e_cbrt4']
        rho_n_cbrt = cupy.cbrt(rho_n)
        beta = rho_e_cbrt + rho_n_cbrt
        beta2 = cupy.square(beta)
        beta3 = beta * beta2
        beta5 = beta2 * beta3
        beta6 = beta3 * beta3
        denom = a - b * beta3 + c * beta6
        denom2 = cupy.square(denom)

        exc = -rho_e / denom
        numer_n = a * rho_e - b * rho_e_cbrt4 * beta2 \
                + c * cupy.multiply(rho_e * beta5,
                                    rho_e_cbrt - rho_n_cbrt)
        vxc_n = -numer_n / denom2
        numer_e = a * rho_n - b * rho_n_cbrt**4 * beta2 \
                + c * cupy.multiply(rho_n * beta5,
                                    rho_n_cbrt - rho_e_cbrt)
        vxc_e = -numer_e / denom2
    return exc, vxc_n, vxc_e


def _get_epc_vmat_task(mf, grids, grids_n, sorted_mol_e, dm_e, opt_n, mol_n_all,
                       dm_n_all, n_slices, device_id=0):
    with cupy.cuda.Device(device_id):
        dm_e = cupy.asarray(dm_e)
        dm_n_all = cupy.asarray(dm_n_all)

        sorted_mol_n = opt_n._sorted_mol
        ao_idx_n = cupy.asarray(opt_n._ao_idx)
        non0ao_idx_n = grids_n.get_non0ao_idx(opt_n)

        nao_e = sorted_mol_e.nao
        vxc_e = cupy.zeros((nao_e, nao_e))
        vxc_n = cupy.zeros((mol_n_all.nao, mol_n_all.nao))
        exc_sum = cupy.asarray(0.0)

        grid_start, grid_end = numint.gen_grid_range(grids.size, device_id)
        p1 = 0
        for ao_e, idx_e, weight, coords in mf.components['e']._numint.block_loop(
                sorted_mol_e, grids, nao_e,
                grid_range=(grid_start, grid_end)):
            p0, p1 = p1, p1 + weight.size
            if len(idx_e) == 0:
                continue
            dm_e_mask = dm_e[idx_e[:,None],idx_e]
            rho_e = numint.eval_rho(sorted_mol_e, ao_e, dm_e_mask, hermi=1)
            rho_e = cupy.maximum(rho_e, 0)
            common = precompute_epc_electron(mf.epc, rho_e)

            vxc_e_grid = 0
            has_epc = False
            block_id = (grid_start + p0) // numint.MIN_BLK_SIZE
            pad, idx_n, non0shl_idx, ctr_offsets_slice, ao_loc_slice = non0ao_idx_n[block_id]
            if len(idx_n) > 0:
                ao_n = numint.eval_ao(
                    sorted_mol_n, coords, deriv=0,
                    screening_info=(len(idx_n), non0shl_idx, ao_loc_slice, ctr_offsets_slice),
                    gdftopt=opt_n,
                    transpose=False)
                if pad > 0:
                    ao_n[-pad:,:] = 0.0
                orig_idx_n = ao_idx_n[idx_n]
                # TODO: Evaluate component-local nuclear densities and build
                # their EPC matrices in grouped kernels without a padded DM.
                for n_type in n_slices:
                    n0, n1 = n_slices[n_type]
                    mask_n = (orig_idx_n >= n0) & (orig_idx_n < n1)
                    # Compact the component AO indices once for both arrays.
                    idx = cupy.where(mask_n)[0]
                    idx_n_t = idx_n[idx]
                    if idx_n_t.size == 0:
                        continue
                    ao_n_t = ao_n[idx]
                    dm_n_mask = dm_n_all[idx_n_t[:,None],idx_n_t]
                    rho_n = numint.eval_rho(sorted_mol_n, ao_n_t, dm_n_mask, hermi=1)
                    rho_n = cupy.maximum(rho_n, 0)

                    exc, vxc_n_grid, vxc_e_grid_t = eval_epc(common, rho_n)
                    vxc_e_grid += vxc_e_grid_t
                    exc_sum += cupy.dot(rho_n * weight, exc)
                    has_epc = True

                    aow_n = numint._scale_ao(ao_n_t, 0.5 * weight * vxc_n_grid)
                    add_sparse(vxc_n, ao_n_t.dot(aow_n.T), idx_n_t)

            if has_epc:
                aow_e = numint._scale_ao(ao_e, 0.5 * weight * vxc_e_grid)
                add_sparse(vxc_e, ao_e.dot(aow_e.T), idx_e)

    return vxc_e, vxc_n, exc_sum.real.item()


def _hash_grids(grids):
    return hash((
            grids.level,
            grids.atom_grid if not isinstance(grids.atom_grid, dict) else
            tuple(grids.atom_grid.items()),
            grids.coords.shape if grids.coords is not None else None,
            grids.coords[0,0].item() if grids.coords is not None else None,
            grids.coords[-1,-1].item() if grids.coords is not None else None,
            grids.weights.size if grids.weights is not None else None,
            grids.weights[0].item() if grids.weights is not None else None,
            grids.weights[-1].item() if grids.weights is not None else None))


class InteractionCorrelation(hf.InteractionCoulomb):
    def __init__(self, *args, epc=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.epc = epc
        self.grids = None
        self._elec_grids_hash = None
        self._skip_epc = False

    def _need_epc(self):
        if self.epc is None:
            return False
        if self.mf1_type == 'e':
            if self.mf2_type.startswith('n'):
                if self.mf2.mol.super_mol.atom_pure_symbol(self.mf2.mol.atom_index) == 'H':
                    if isinstance(self.epc, str) or \
                            self.mf2.mol.atom_index in self.epc['epc_nuc']:
                        symbol = self.mf2.mol.super_mol.atom_symbol(self.mf2.mol.atom_index)
                        if 'H+' in symbol or 'H*' in symbol or 'H#' in symbol:
                            warnings.warn('Hydrogen isotopes detected. Are you sure you want epc?')
                        return True
        if self.mf2_type == 'e':
            if self.mf1_type.startswith('n'):
                if self.mf1.mol.super_mol.atom_pure_symbol(self.mf1.mol.atom_index) == 'H':
                    if isinstance(self.epc, str) or \
                            self.mf1.mol.atom_index in self.epc['epc_nuc']:
                        symbol = self.mf1.mol.super_mol.atom_symbol(self.mf1.mol.atom_index)
                        if 'H+' in symbol or 'H*' in symbol or 'H#' in symbol:
                            warnings.warn('Hydrogen isotopes detected. Are you sure you want epc?')
                        return True
        return False


def _get_epc_vmat(mf, dm):
    def no_epc():
        return {t: 0 for t in mf.components}

    if mf.epc is None:
        return no_epc()

    mf_e = mf.components['e']
    mol_e = mf_e.mol
    dm_e = cupy.asarray(dm['e'])
    if isinstance(mf_e, scf.uhf.UHF):
        assert dm_e.ndim > 2 and dm_e.shape[0] == 2
        dm_e = dm_e[0] + dm_e[1]

    grids_e = mf_e.grids
    if grids_e.coords is None:
        rks.initialize_grids(mf_e, mol_e, dm_e)
    elec_grids_hash = _hash_grids(grids_e)
    grids_changed = (mf._elec_grids_hash != elec_grids_hash)
    if grids_changed and mf._epc_n_types is not None:
        if len(mf._epc_n_types) > 0:
            mf._skip_epc = False
    if mf._skip_epc:
        return no_epc()

    if mf._epc_n_types is None:
        n_types = []
        for t_pair, interaction in mf.interactions.items():
            if interaction._need_epc():
                if t_pair[0].startswith('n'):
                    n_type  = t_pair[0]
                else:
                    n_type  = t_pair[1]
                n_types.append(n_type)
        mf._epc_n_types = n_types
    else:
        n_types = mf._epc_n_types

    if len(n_types) == 0:
        mf._skip_epc = True
        return no_epc()

    mol_n_all = mf.components[n_types[0]].mol
    n_slices = {}
    p0 = 0
    for n_type in n_types:
        mol_n_t = mf.components[n_type].mol
        if n_type != n_types[0]:
            mol_n_all = gto.conc_mol(mol_n_all, mol_n_t)
        p1 = p0 + mol_n_t.nao
        n_slices[n_type] = (p0, p1)
        p0 = p1

    ni_n = numint.NumInt()
    ni_n.build(mol_n_all, grids_e.coords)
    if mf.grids is None or grids_changed:
        grids = copy.copy(grids_e)
        grids.mol = mol_n_all
        grids._non0ao_idx = None
        non0ao_idx_n = grids.get_non0ao_idx(ni_n.gdftopt)
        block_ids = [i for i, x in enumerate(non0ao_idx_n) if len(x[1]) > 0]
        if len(block_ids) == 0:
            mf._skip_epc = True
            return no_epc()
        starts = (cupy.asarray(block_ids)[:,None] * numint.MIN_BLK_SIZE
                  + cupy.arange(numint.MIN_BLK_SIZE))
        valid_idx = starts[starts < grids_e.coords.shape[0]]
        mf.grids = copy.copy(grids_e)
        mf.grids.coords = grids_e.coords[valid_idx]
        mf.grids.weights = grids_e.weights[valid_idx]
        if getattr(grids_e, 'atm_idx', None) is not None:
            mf.grids.atm_idx = grids_e.atm_idx[valid_idx]
        if getattr(grids_e, 'quadrature_weights', None) is not None:
            mf.grids.quadrature_weights = grids_e.quadrature_weights[valid_idx]
        mf.grids.non0tab = None
        mf.grids.screen_index = None
        mf.grids._non0ao_idx = None
        mf._elec_grids_hash = elec_grids_hash
    grids = mf.grids

    ni_e = mf_e._numint
    if ni_e.gdftopt is None:
        ni_e.build(mol_e, grids.coords)
    opt_e = ni_e.gdftopt
    sorted_mol_e = opt_e._sorted_mol
    dm_e = opt_e.sort_orbitals(dm_e, axis=[0,1])

    opt_n = ni_n.gdftopt
    grids_n = copy.copy(grids)
    grids_n.mol = mol_n_all
    grids_n._non0ao_idx = None

    dm_n_all = cupy.zeros((mol_n_all.nao, mol_n_all.nao))
    for n_type in n_types:
        p0, p1 = n_slices[n_type]
        dm_n_all[p0:p1,p0:p1] = cupy.asarray(dm[n_type])
    dm_n_all = opt_n.sort_orbitals(dm_n_all, axis=[0,1])

    grids.get_non0ao_idx(opt_e)
    grids_n.get_non0ao_idx(opt_n)
    release_gpu_stack()
    cupy.cuda.get_current_stream().synchronize()
    futures = []
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        for device_id in range(num_devices):
            future = executor.submit(
                _get_epc_vmat_task, mf, grids, grids_n, sorted_mol_e,
                dm_e, opt_n, mol_n_all, dm_n_all, n_slices, device_id)
            futures.append(future)

    vxc_e_dist = []
    vxc_n_dist = []
    exc_sum = 0
    for future in futures:
        vxc_e_t, vxc_n_t, exc_sum_t = future.result()
        vxc_e_dist.append(vxc_e_t)
        vxc_n_dist.append(vxc_n_t)
        exc_sum += exc_sum_t
    vxc_e = reduce_to_device(vxc_e_dist, inplace=True)
    vxc_n = reduce_to_device(vxc_n_dist, inplace=True)

    vxc_e = vxc_e + vxc_e.conj().T
    vxc_e = opt_e.unsort_orbitals(vxc_e, axis=[0,1])
    vxc_n = vxc_n + vxc_n.conj().T
    vxc_n = opt_n.unsort_orbitals(vxc_n, axis=[0,1])

    epc = {}
    epc['e'] = tag_array(vxc_e, exc=exc_sum)
    for t in mf.components:
        if t == 'e':
            continue
        if t in n_types:
            p0, p1 = n_slices[t]
            epc[t] = tag_array(vxc_n[p0:p1,p0:p1], exc=0)
        else:
            epc[t] = 0
    return epc


class KS(hf.HF):
    _keys = hf.HF._keys.union({'xc_e', 'epc'})

    def __init__(self, mol, *args, xc=None, epc=None, **kwargs):
        super().__init__(mol, *args, **kwargs)
        if xc is None:
            raise RuntimeError('Please provide electronic xc via "xc" kwarg!')
        self.xc_e = xc
        self.epc = epc

        for t, comp in self.mol.components.items():
            if t.startswith('n'):
                if self.epc is None:
                    mf = scf.RHF(comp)
                else:
                    mf = dft.RKS(comp, xc='HF')
                self.components[t] = hf.general_scf(mf,
                                                    charge=-1. * self.mol.atom_charge(comp.atom_index),
                                                    mass=self.mol.mass[comp.atom_index] * nist.ATOMIC_MASS
                                                         / nist.E_MASS,
                                                    is_nucleus=True, nuc_occ_state=0)
            else:
                if self.unrestricted:
                    if self.epc is None and self.xc_e.upper() == 'HF':
                        mf = scf.UHF(comp)
                    else:
                        mf = dft.UKS(comp, xc=self.xc_e)
                else:
                    if getattr(comp, 'nhomo', None) is not None or comp.spin != 0:
                        if self.epc is None and self.xc_e.upper() == 'HF':
                            mf = scf.UHF(comp)
                        else:
                            mf = dft.UKS(comp, xc=self.xc_e)
                    else:
                        if self.epc is None and self.xc_e.upper() == 'HF':
                            mf = scf.RHF(comp)
                        else:
                            mf = dft.RKS(comp, xc=self.xc_e)
                charge = 1.
                if t.startswith('p'):
                    charge = -1.
                self.components[t] = hf.general_scf(mf, charge=charge)
        self.interactions = hf_cpu.generate_interactions(
            self.components, InteractionCorrelation,
            self.max_memory, self.direct_scf_tol, epc=self.epc)
        #####
        self._epc_n_types = None
        self._skip_epc = False
        if isinstance(self.components['e'], scf.hf.KohnShamDFT):
            self._numint = self.components['e']._numint
        else:
            self._numint = None
        self.grids = None
        self._elec_grids_hash = None

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None: dm = self.make_rdm1()
        if h1e is None: h1e = self.get_hcore()
        if vhf is None: vhf = self.get_veff(self.mol, dm)
        self.scf_summary['e1'] = 0
        self.scf_summary['e2'] = 0
        self.scf_summary['coul'] = 0
        self.scf_summary['exc'] = 0
        e_elec = 0
        e2 = 0
        nuclear_components = {t: comp for t, comp in self.components.items()
                              if t.startswith('n')}
        for t, comp in self.components.items():
            if t.startswith('n'):
                continue
            e_elec_t, e2_t = comp.energy_elec(dm[t], h1e[t], vhf[t])
            e_elec += e_elec_t
            e2 += e2_t
            self.scf_summary['e1'] += comp.scf_summary['e1']
            self.scf_summary['e2'] += comp.scf_summary['e2']
            if hasattr(vhf[t], 'exc'):
                self.scf_summary['coul'] += comp.scf_summary['coul']
                self.scf_summary['exc'] += comp.scf_summary['exc']
            elif 'e2' in comp.scf_summary:
                self.scf_summary['coul'] += comp.scf_summary['e2']
        if nuclear_components:
            e_elec_n, e2_n = hf._grouped_energy(nuclear_components, dm, h1e, vhf)
            e_elec += e_elec_n
            e2 += e2_n
            for t in nuclear_components:
                comp = nuclear_components[t]
                self.scf_summary['e1'] += comp.scf_summary['e1']
                self.scf_summary['e2'] += comp.scf_summary['e2']
                if hasattr(vhf[t], 'exc'):
                    self.scf_summary['coul'] += comp.scf_summary['coul']
                    self.scf_summary['exc'] += comp.scf_summary['exc']
                elif 'e2' in comp.scf_summary:
                    self.scf_summary['coul'] += comp.scf_summary['e2']
        return e_elec, e2

    def _get_vint_fast(self, mol=None, dm=None, dm_last=None, vhf_last=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()

        vint = super()._get_vint(mol, dm, dm_last, vhf_last,
                                 coulomb_only=True)
        epc = _get_epc_vmat(self, dm)
        for t in vint:
            vint_inc = getattr(vint[t], 'vint_inc', 0)
            coul_vint = cupy.asarray(vint[t])
            vint[t] = tag_array(coul_vint + epc[t], vint_inc=vint_inc)
            if hasattr(epc[t], 'exc'):
                vint[t] = tag_array(vint[t], exc=epc[t].exc, vj=coul_vint)
        return vint

    _get_vint = _get_vint_fast

    def copy(self):
        new = scf.hf.SCF.copy(self)
        if hasattr(self, 'f') and self.f is not None:
            new.f = cupy.array(self.f, copy=True)

        new.components = {}
        for t, comp in self.components.items():
            new.components[t] = hf.general_scf(comp.undo_component().copy(),
                                               charge=comp.charge,
                                               mass=comp.mass,
                                               is_nucleus=comp.is_nucleus,
                                               nuc_occ_state=comp.nuc_occ_state)

        new.interactions = hf_cpu.generate_interactions(
            new.components, InteractionCorrelation,
            new.max_memory, new.direct_scf_tol, epc=new.epc)

        if isinstance(new.components['e'], scf.hf.KohnShamDFT):
            new._numint = new.components['e']._numint
        else:
            new._numint = None
        new.grids = None
        new._elec_grids_hash = None
        new._epc_n_types = None
        new._skip_epc = False
        return new

    def reset(self, mol=None):
        '''Reset mol and relevant attributes associated to the old mol object'''
        if mol is not None:
            self.mol = mol
        scf.hf.SCF.reset(self, mol=mol) # do not call neo.HF.reset
        components = self.components.copy()
        if components.keys() != self.mol.components.keys():
            self.mo_coeff = None
        self.components.clear()
        for t, comp in self.mol.components.items():
            is_nucleus = t.startswith('n')
            unrestricted = not is_nucleus and (self.unrestricted or comp.spin != 0 or
                                               getattr(comp, 'nhomo', None) is not None)
            mf = components.get(t)
            # Preserve the existing HF/KS method; select a method only for new components.
            if mf is None:
                is_dft = self.epc is not None or not is_nucleus and self.xc_e.upper() != 'HF'
            else:
                is_dft = isinstance(mf, scf.hf.KohnShamDFT)
            if is_dft:
                mf_class = dft.uks.UKS if unrestricted else dft.rks.RKS
            else:
                mf_class = scf.uhf.UHF if unrestricted else scf.hf.RHF
            if isinstance(mf, mf_class):
                mf.reset(comp)
            else:
                if is_dft:
                    xc = mf.xc if mf is not None else ('HF' if is_nucleus else self.xc_e)
                    mf = mf_class(comp, xc=xc)
                else:
                    mf = mf_class(comp)
                self.mo_coeff = None
            if is_nucleus:
                self.components[t] = hf.general_scf(mf,
                                                    charge=-1. * self.mol.atom_charge(comp.atom_index),
                                                    mass=self.mol.mass[comp.atom_index] * nist.ATOMIC_MASS
                                                         / nist.E_MASS,
                                                    is_nucleus=True,
                                                    nuc_occ_state=getattr(mf, 'nuc_occ_state', 0))
            else:
                charge = -1. if t.startswith('p') else 1.
                self.components[t] = hf.general_scf(mf, charge=charge)
        # Recreate pair molecules, spin flags and EPC state after class selection.
        self.interactions.clear()
        self.interactions.update(hf_cpu.generate_interactions(
            self.components, InteractionCorrelation,
            self.max_memory, self.direct_scf_tol, epc=self.epc))
        # EPC grids
        self._epc_n_types = None
        self._skip_epc = False
        if isinstance(self.components['e'], scf.hf.KohnShamDFT):
            self._numint = self.components['e']._numint
        else:
            self._numint = None
        self.grids = None
        self._elec_grids_hash = None
        return self

    def to_cpu(self):
        obj = ks_cpu.KS(self.mol, unrestricted=self.unrestricted, xc=self.xc_e, epc=self.epc)
        for key in self._keys:
            if key in ('components', 'interactions'):
                continue
            if hasattr(self, key):
                setattr(obj, key, hf._to_cpu(getattr(self, key)))
        obj.components = {t: comp.to_cpu() for t, comp in self.components.items()}
        obj.interactions = hf_cpu.generate_interactions(
            obj.components, ks_cpu.InteractionCorrelation,
            obj.max_memory, obj.direct_scf_tol, epc=obj.epc)
        if isinstance(obj.components['e'], scf_cpu.hf.KohnShamDFT):
            obj._numint = obj.components['e']._numint
        else:
            obj._numint = None
        obj.grids = None
        obj._elec_grids_hash = None
        obj._epc_n_types = None
        obj._skip_epc = False
        return obj

    to_gpu = utils.to_gpu


def from_cpu(mf):
    out = KS(mf.mol, unrestricted=mf.unrestricted, xc=mf.xc_e, epc=mf.epc)
    for key, val in mf.__dict__.items():
        if key in ('components', 'interactions', 'grids', '_elec_grids_hash',
                   '_epc_n_types', '_skip_epc', '_numint'):
            continue
        setattr(out, key, hf._to_gpu(val))
    out.components = {t: comp.to_gpu() for t, comp in mf.components.items()}
    # The parent KS object shares the electronic component's NumInt object.
    # Do not carry over the CPU parent _numint cache.
    if isinstance(out.components['e'], scf.hf.KohnShamDFT):
        out._numint = out.components['e']._numint
    else:
        out._numint = None
    out.grids = None
    out._elec_grids_hash = None
    out._epc_n_types = None
    out._skip_epc = False
    out.interactions = hf_cpu.generate_interactions(
        out.components, InteractionCorrelation,
        out.max_memory, out.direct_scf_tol, epc=out.epc)
    return out
