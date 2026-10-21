'''Analytic gradients for multicomponent density fitting.'''

import ctypes
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.df import addons

from gpu4pyscf.df.grad import rhf as df_rhf_grad
from gpu4pyscf.gto.mole import _split_l_ctr_groups
from gpu4pyscf.df.int3c2e_bdiv import int2c2e, int2c2e_ip1_per_atom, int3c2e_scheme
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.neo import df as neo_df
from gpu4pyscf.neo import grad, int3c2e_bdiv


class _ElectronicGradWithoutJ:
    '''Electronic component gradient with J supplied by the global DF path.'''

    def jk_energy_per_atom(self, dm=None, j_factor=1, k_factor=1, *,
                           omega=None, lr_factor=None, sr_factor=None,
                           hermi=0, verbose=None):
        if k_factor == 0:
            return np.zeros((self.mol.natm, 3))
        # Global J does not build the electronic DF object. Mixed RSH K
        # builds only its range-separated copy, leaving this auxmol unset.
        with_df = self.base.with_df
        if with_df.auxmol is None:
            with_df.auxmol = addons.make_auxmol(with_df.mol, with_df.auxbasis)
        return super().jk_energy_per_atom(
            dm, 0, k_factor, omega=omega, lr_factor=lr_factor,
            sr_factor=sr_factor, hermi=hermi, verbose=verbose)


def _j_energy_per_atom(int3c2e_opt, dm, hermi=0, auxbasis_response=True,
                       verbose=None, *, charges, df_nn=False):
    """Evaluate the global NEO DF-J contribution to the gradient.

    ``charges`` and ``df_nn`` define the NEO Coulomb energy expression.
    """
    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    components = int3c2e_opt.components

    # NEO Coulomb interactions are spin-insensitive, so unrestricted
    # electronic densities are spin-summed before the component contraction.
    dms = {}
    for t, dm_t in dm.items():
        dm_t = cp.asarray(dm_t)
        if dm_t.ndim == 3:
            assert t == 'e' and dm_t.shape[0] == 2
            dm_t = dm_t[0] + dm_t[1]
        dms[t] = dm_t

    # contract_dm returns the component-local sorted DMs used by the
    # component-aware derivative kernel, avoiding a second AO transformation.
    rho, dms, local_ao_loc = int3c2e_opt.contract_dm(dms, hermi, return_transformed_dm=True)
    t0 = log.timer_debug1('contract dm', *t0)
    j2c = int2c2e(auxmol)

    # The common auxiliary metric is applied to all component source vectors.
    rho = auxmol.mat_dot_C(cp.vstack([rho[t] for t in int3c2e_opt.component_names]))
    if mol.omega <= 0 and not auxmol.mol.cart:
        rho = df_rhf_grad._gen_metric_solver(j2c, 'CD')(rho.T).T
    else:
        rho = df_rhf_grad._gen_metric_solver(j2c, 'ED')(rho.T).T
    j2c = None
    if auxbasis_response:
        rho = auxmol.mat_dot_CT(rho)
    rho = {t: rho[i] for i, t in enumerate(int3c2e_opt.component_names)}
    # Assemble the source seen by each distinguishable component. Nuclear
    # self-density is excluded; nuclear-nuclear sources are included with df_nn.
    rhoj = neo_df._get_rhoj(
        rho['e'], {t: rho[t] for t in components if t != 'e'},
        charges, components, df_nn)
    # Apply the target charge to the smaller component DM; _get_rhoj has
    # already applied source charges to the auxiliary vectors.
    for t in int3c2e_opt.component_names:
        if charges[t] != 1:
            dms[t] *= charges[t]
    if auxbasis_response:
        auxvec = rhoj
    else:
        auxvec = auxmol.mat_dot_CT(cp.vstack([rhoj[t] for t in int3c2e_opt.component_names]))
        auxvec = {t: auxvec[i] for i, t in enumerate(int3c2e_opt.component_names)}

    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
        short_range=mol.omega<0, gout_width=54, deriv=(1,0,0))
    lmax = mol.uniq_l_ctr[:,0].max()
    laux = auxmol.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
    bas_ij_idx, shl_pair_offsets = int3c2e_bdiv._aggregate_shl_pair_blocks(
        mol, int3c2e_opt.bas_ij_blocks, nsp_per_block[0]*16)
    # Match the derivative kernel's 16-shell auxiliary capacity.
    _, aux_counts = _split_l_ctr_groups(
        auxmol.uniq_l_ctr, auxmol.l_ctr_counts, [16]*(laux+1))
    ksh_offsets_cpu = np.append(0, np.cumsum(aux_counts))
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = df_rhf_grad.libvhf_rys.sum_j_int3c2e_ip1_multi_in
    ej = cp.zeros((mol.natm, 3))
    if auxbasis_response:
        ej_aux = cp.zeros_like(ej)
        ej_aux_ptr = ctypes.cast(ej_aux.data.ptr, ctypes.c_void_p)
    else:
        ej_aux_ptr = lib.c_null_ptr()

    # Each same-component shell pair selects its dense DM and auxiliary source
    # without constructing a block-diagonal DM or a packed dm*auxvec tensor.
    component_index = {t: i for i, t in enumerate(int3c2e_opt.component_names)}
    pair_component = cp.asarray(np.hstack([
        np.full(len(bas_ij), component_index[t], dtype=np.int32)
        for (_, bas_ij), t in zip(
            int3c2e_opt.bas_ij_blocks, int3c2e_opt.block_components)]))
    dm_ptrs = [dms[t].data.ptr for t in int3c2e_opt.component_names]
    dm_ptrs = cp.asarray(np.asarray(dm_ptrs, dtype=np.uintp))
    auxvec_ptrs = [auxvec[t].data.ptr for t in int3c2e_opt.component_names]
    auxvec_ptrs = cp.asarray(np.asarray(auxvec_ptrs, dtype=np.uintp))
    local_ao_loc = cp.asarray(local_ao_loc, dtype=np.int32)
    component_nao = [dms[t].shape[-1] for t in int3c2e_opt.component_names]
    component_nao = cp.asarray(component_nao, dtype=np.int32)

    err = kern(
        ctypes.cast(ej.data.ptr, ctypes.c_void_p), ej_aux_ptr,
        ctypes.cast(dm_ptrs.data.ptr, ctypes.c_void_p),
        ctypes.cast(auxvec_ptrs.data.ptr, ctypes.c_void_p),
        ctypes.byref(int3c2e_envs),
        ctypes.c_int(shm_size_max),
        ctypes.c_int(len(shl_pair_offsets) - 1),
        ctypes.c_int(len(ksh_offsets_cpu) - 1),
        ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(ksh_offsets_gpu.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
        ctypes.cast(pair_component.data.ptr, ctypes.c_void_p),
        ctypes.cast(local_ao_loc.data.ptr, ctypes.c_void_p),
        ctypes.cast(component_nao.data.ptr, ctypes.c_void_p),
        ctypes.c_int(mol.natm))
    if err != 0:
        raise RuntimeError('int3c2e_ejk_ip1 failed')
    ej *= 2

    natm = components['e'].natm
    de = cp.zeros((natm, 3))
    # Each component repeats the physical atoms; sum matching atom rows.
    p0 = 0
    for t in int3c2e_opt.component_names:
        p1 = p0 + components[t].natm
        de += ej[p0:p1]
        p0 = p1
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)
    if auxbasis_response:
        ej_aux *= 2
        rho_total = sum(charges[t] * rho[t] for t in int3c2e_opt.component_names)
        # Remove nuclear self-products excluded from the NEO Coulomb energy.
        if df_nn:
            dm_aux = rho_total[:,None] * rho_total
            # Rows hold nuclear auxiliary densities; subtract all self-products
            # in one contraction instead of one auxiliary-square array per nucleus.
            nuclear_keys = [t for t in int3c2e_opt.component_names if t != 'e']
            if nuclear_keys:
                rho_n = cp.empty((len(nuclear_keys), rho_total.size), dtype=rho_total.dtype)
                for i, t in enumerate(nuclear_keys):
                    cp.multiply(rho[t], charges[t], out=rho_n[i])
                dm_aux = contract('ir,is->rs', rho_n, rho_n, alpha=-1, beta=1, out=dm_aux)
        else:
            rho_n = sum(charges[t] * rho[t] for t in int3c2e_opt.component_names if t != 'e')
            dm_aux = rho_total[:,None] * rho_total - rho_n[:,None] * rho_n
        ej_aux[:components['e'].natm] -= cp.asarray(int2c2e_ip1_per_atom(auxmol, dm_aux))
        # The first natm auxiliary-gradient rows correspond to physical atoms.
        de += ej_aux[:natm]
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)
    return de.get()


def grad_int(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_occ is None:
        mo_occ = mf.mo_occ
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff

    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)
    dm = mf.make_rdm1(mo_coeff, mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)

    with_df = mf.with_df
    if with_df._auxmol_atom_major is None:
        with_df._auxmol_atom_major = with_df.make_auxmol_atom_major()
    auxmol = with_df._auxmol_atom_major
    with_df.reset() # Release GPU memory
    int3c2e_opt = int3c2e_bdiv.Int3c2eOpt(mf.mol.components, auxmol).build()
    charges = {t: mf.components[t].charge for t in mf.components}
    de = _j_energy_per_atom(int3c2e_opt, dm, hermi=1,
                            auxbasis_response=mf_grad.auxbasis_response,
                            verbose=mf_grad.verbose, charges=charges,
                            df_nn=with_df.df_nn)
    if not mf.with_df.df_nn:
        # Nuclear-nuclear J omitted from DF is differentiated with exact ERIs.
        eri_interactions = [(pair, interaction) for pair, interaction in mf.interactions.items()
                            if 'e' not in pair]
        if eri_interactions:
            de += grad.grad_eri(mf_grad, dm, eri_interactions, range(mol.natm))
    de = de[list(atmlst)]
    if log.verbose >= logger.DEBUG:
        log.debug('gradients of Coulomb interaction')
        grad.rhf_grad_cpu._write(log, mol, de, atmlst)
    return de


class Gradients(grad.Gradients):
    '''Analytic gradient for multicomponent density fitting.'''

    auxbasis_response = True
    grad_int = grad_int

    def __init__(self, mf):
        super().__init__(mf)
        comp = self.components['e']
        self.components['e'] = comp.view(lib.make_class(
            (_ElectronicGradWithoutJ, comp.__class__)))

    def reset(self, mol=None):
        super().reset(mol)
        comp = self.components['e']
        if not isinstance(comp, _ElectronicGradWithoutJ):
            self.components['e'] = comp.view(lib.make_class(
                (_ElectronicGradWithoutJ, comp.__class__)))
        return self


Grad = Gradients
