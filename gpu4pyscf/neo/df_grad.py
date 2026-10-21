'''Analytic gradients for multicomponent density fitting.'''

import ctypes
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.gto import ATOM_OF
from pyscf.df import addons
from cupyx.scipy.linalg import solve_triangular

from gpu4pyscf.df.grad import rhf as df_rhf_grad
from gpu4pyscf.gto.mole import SortedMole, _split_l_ctr_groups
from gpu4pyscf.df.int3c2e_bdiv import (
    int2c2e, int2c2e_ip1_per_atom, int3c2e_scheme, _split_l_ctr_pattern)
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, cholesky, get_avail_mem, ndarray
from gpu4pyscf.lib.utils import splits_by_blocksize
from gpu4pyscf.neo import df as neo_df
from gpu4pyscf.neo import grad, int3c2e_bdiv

class _ElectronicGradWithoutJ:
    '''Compute electronic K; the NEO interaction builder computes global J.'''

    def __init__(self, comp, *, with_full_range_k):
        self.__dict__.update(comp.__dict__)
        # The top-level assembly assigns full-range K to this worker or joint J/K.
        self.with_full_range_k = with_full_range_k

    def jk_energy_per_atom(self, dm=None, j_factor=1, k_factor=1, *,
                           omega=None, lr_factor=None, sr_factor=None,
                           hermi=0, verbose=None):
        if k_factor == 0:
            return np.zeros((self.mol.natm, 3))
        if (omega is None or omega == 0) and not self.with_full_range_k:
            return np.zeros((self.mol.natm, 3))
        # Range-separated K can initialize only its operator-specific DF.
        # Prepare the full-range auxiliary molecule for derivative calls.
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


def _jk_energy_per_atom(mf_grad, dm, k_factor, hermi, verbose):
    """Joint global-J/electronic-K gradient.

    Adapted from df.grad.rhf/uhf._jk_energy_per_atom. Embedding their internal
    forward and derivative loops lets global J consume the same integral tiles.
    NEO adds component-local J sources; K uses electronic AOs/auxiliaries and
    the electronic metric, while J uses the combined auxiliary metric.
    """
    mf = mf_grad.base
    dm_e = dm['e']
    with_df = mf.with_df
    assert hermi == 1
    log = logger.new_logger(mf, verbose)
    t0 = log.init_timer()
    auxmol = with_df.make_auxmol()
    e_df = mf.components['e'].with_df
    auxmol_e = addons.make_auxmol(mf.mol.components['e'], e_df.auxbasis)
    naux_e = auxmol_e.nao
    natm = mf.mol.natm
    # Preserve the electronic auxiliary prefix; fold ATOM_OF to physical atoms.
    auxmol._bas[:,ATOM_OF] %= natm
    e_indices = cp.arange(naux_e)
    n_indices = cp.arange(naux_e, auxmol.nao)
    auxmol._atm = auxmol._atm[:natm]
    auxmol._atom = auxmol._atom[:natm]
    with_df.reset()
    e_df.reset()
    e_df.auxmol = auxmol_e
    components = mf.mol.components.copy()
    components['e'] = SortedMole.from_mol(components['e'], decontract=True)
    intopt = int3c2e_bdiv.Int3c2eOpt(components, auxmol).build()
    mol = intopt.mol
    auxmol = intopt.auxmol
    # Split homogeneous groups at the electronic/nuclear auxiliary boundary.
    aux_l_ctr = auxmol.uniq_l_ctr, auxmol.l_ctr_counts
    e_shell = auxmol.get_orig_shell_idx() < auxmol_e.nbas
    shell_offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
    repeats = []
    counts = []
    for s0, s1 in zip(shell_offsets[:-1], shell_offsets[1:]):
        ne = np.count_nonzero(e_shell[s0:s1])
        sizes = [n for n in (ne, s1-s0-ne) if n]
        counts.extend(sizes)
        repeats.append(len(sizes))
    auxmol.uniq_l_ctr = np.repeat(auxmol.uniq_l_ctr, repeats, axis=0)
    auxmol.l_ctr_counts = np.asarray(counts)
    # Match the standalone J derivative's 16-shell auxiliary groups.
    auxmol.uniq_l_ctr, auxmol.l_ctr_counts = _split_l_ctr_groups(
        auxmol.uniq_l_ctr, auxmol.l_ctr_counts, [16]*(auxmol.uniq_l_ctr[:,0].max()+1))
    keys = intopt.component_names
    component_mols, local_ao_loc = intopt._component_mol_data()
    dm_factor_l, dm_factor_r = df_rhf_grad._factorize_dm(component_mols['e'], dm_e, hermi)
    unrestricted = dm_e.ndim == 3
    nao, nocc = dm_factor_l.shape[-2:]

    # Map same-component pairs to local Cartesian density matrices.
    global_to_local = np.empty(mol.nao, dtype=np.int32)
    for ish in range(mol.nbas):
        p0, p1 = mol.ao_loc[ish:ish+2]
        global_to_local[p0:p1] = local_ao_loc[ish] + np.arange(p1-p0)
    global_to_local = cp.asarray(global_to_local)
    pairs = intopt.pair_and_diag_indices(cart=True, original_ao_order=False)
    pair_addresses = {}
    dms = {}
    weights = []
    pair_offsets = [0]
    groups = []
    for ic, t in enumerate(keys):
        addresses, diags = pairs[t]
        i, j = divmod(addresses, mol.nao)
        nao_t = component_mols[t].nao
        pair_addresses[t] = global_to_local[i]*nao_t + global_to_local[j]
        dm_t = dm[t]
        if dm_t.ndim == 3:
            dm_t = dm_t[0] + dm_t[1]
        dms[t] = component_mols[t].apply_C_mat_CT(dm_t)
        packed = dms[t].ravel()[pair_addresses[t]].copy()
        packed *= 2
        packed[diags] *= .5
        weights.append(packed)
        pair_offsets.append(pair_offsets[-1] + len(packed))
        if ic > 0:
            if groups and groups[-1][2] == len(packed):
                groups[-1][1] = ic+1
            else:
                groups.append([ic, ic+1, len(packed)])
    weights = cp.hstack(weights)
    pair_addresses_e = pair_addresses['e']
    i_addr, j_addr = divmod(pair_addresses_e, nao)
    npair_e = len(pair_addresses_e)
    npair = pair_offsets[-1]
    t0 = log.timer_debug1('NEO shared J/K prepare', *t0)

    # Begin df.grad.rhf/uhf._jk_energy_per_atom forward occupied-pair transforms.
    # NEO delta: only electronic auxiliary columns enter K's tensor.
    naux = auxmol.nao
    mem_free = get_avail_mem(exclude_memory_pool=True)
    spin_sets = 2 if unrestricted else 1
    mem_avail = mem_free - spin_sets*naux*nocc**2*8 - nao**2*8
    batch_size = max(1, min(naux, int(mem_avail*.5/(npair*8))))
    # _split_l_ctr_pattern keeps at least one complete Cartesian shell.
    # Its largest shell must fit even when the memory estimate gives one AO.
    laux = auxmol.uniq_l_ctr[:,0].max()
    batch_size = max(batch_size, (laux+1)*(laux+2)//2)
    eval_j3c, aux_sorting, _, aux_offsets = intopt.int3c2e_evaluator(
        aux_batch_size=batch_size, reorder_aux=True, cart=True)
    batch_size = min(batch_size, int((aux_offsets[1:]-aux_offsets[:-1]).max()))
    aux_sorting_cpu = cp.asnumpy(aux_sorting)
    # Shell provenance identifies electronic Cartesian auxiliary rows.
    e_cart = np.hstack([np.arange(auxmol.ao_loc[i], auxmol.ao_loc[i+1])
                        for i in np.flatnonzero(e_shell)])
    e_eval = np.sort(aux_sorting_cpu[e_cart])
    e_mask = np.zeros(naux, dtype=bool)
    e_mask[e_eval] = True
    e_ranges = np.flatnonzero(np.diff(np.r_[False, e_mask, False])).reshape(-1, 2)
    aux_coeff = cp.asarray(auxmol.ctr_coeff)
    aux_coeff_eval = cp.empty_like(aux_coeff)
    aux_coeff_eval[aux_sorting] = aux_coeff
    coeff_e = aux_coeff_eval[cp.asarray(e_eval)[:,None], e_indices]
    aux_coeff = aux_coeff_eval = None
    ne_cart = len(e_eval)
    j3c_oo = cp.empty((spin_sets, ne_cart, nocc, nocc))
    rho = cp.empty((len(keys), naux))
    blksize = max(1, min(batch_size, int(mem_avail*.3/(nao*nao*2*8))))
    j3c_full = cp.zeros((nao, nao, blksize))
    buf = cp.empty((npair, batch_size))
    buf1 = cp.empty((blksize, nocc, nao))
    for batch, (a0, a1) in enumerate(zip(aux_offsets[:-1], aux_offsets[1:])):
        compressed = eval_j3c(aux_batch_id=batch, out=buf)
        dk = a1-a0
        for c0, c1, nrow in groups:
            p0, p1 = pair_offsets[c0], pair_offsets[c1]
            contract('np,npq->nq', weights[p0:p1].reshape(c1-c0, nrow),
                     compressed[p0:p1].reshape(c1-c0, nrow, dk),
                     out=rho[c0:c1,a0:a1])
        ecols = np.flatnonzero(e_mask[a0:a1])
        ncols = np.flatnonzero(~e_mask[a0:a1])
        if len(ncols):
            ncols_gpu = cp.asarray(ncols)
            rho[0,a0+ncols_gpu] = weights[:npair_e].dot(compressed[:npair_e,ncols_gpu])
        e0 = np.searchsorted(e_eval, a0)
        for p0, p1 in lib.prange(0, len(ecols), blksize):
            j3c = j3c_full[:,:,:p1-p0]
            # Only electronic columns enter the conventional DFT K transforms.
            columns = cp.asarray(ecols[p0:p1])
            j3c[j_addr,i_addr] = j3c[i_addr,j_addr] = compressed[:npair_e,columns]
            for s in range(spin_sets):
                left = dm_factor_l[s] if unrestricted else dm_factor_l
                right = dm_factor_r[s] if unrestricted else dm_factor_r
                tmp = ndarray((nocc, nao, p1-p0), buffer=buf1)
                contract('pqr,pi->iqr', j3c, right, out=tmp)
                contract('iqr,qj->rij', tmp, left, out=j3c_oo[s,e0+p0:e0+p1])
        if len(ecols):
            rho[0,cp.asarray(e_eval[e0:e0+len(ecols)])] = (
                j3c_oo[:,e0:e0+len(ecols)].trace(axis1=2, axis2=3).sum(axis=0))
    j3c_full = buf = buf1 = compressed = j3c = tmp = eval_j3c = None
    t0 = log.timer_debug1('NEO shared J/K contract dm', *t0)
    # End adapted conventional DFT forward occupied-pair transform block.

    # Two-center kernels use original groups. The global Cholesky's leading
    # block also factors the electronic metric used by K.
    with lib.temporary_env(auxmol, uniq_l_ctr=aux_l_ctr[0], l_ctr_counts=aux_l_ctr[1]):
        j2c = int2c2e(auxmol)
    perm = cp.concatenate((e_indices, n_indices))
    raw_rho = auxmol.mat_dot_C(rho[:,aux_sorting]).T
    low = None
    if not auxmol.mol.cart:
        try:
            low = cholesky(j2c[perm[:,None],perm])
        except RuntimeError:
            pass
    if low is not None:
        rho = solve_triangular(low, raw_rho[perm], lower=True)
        rho = solve_triangular(low.T, rho, lower=False)
        rho_orig = cp.empty_like(rho)
        rho_orig[perm] = rho
        low_e = low[:naux_e,:naux_e]
        metric_e = solve_triangular(low_e, coeff_e.T, lower=True)
        metric_e = solve_triangular(low_e.T, metric_e, lower=False)
    else:
        decompose = 'ED' if auxmol.mol.cart else 'CD'
        rho_orig = df_rhf_grad._gen_metric_solver(j2c, decompose)(raw_rho)
        metric_e = j2c[e_indices[:,None],e_indices]
        metric_e = df_rhf_grad._gen_metric_solver(metric_e, decompose)(coeff_e.T)
    # Follow df.grad.uhf._jk_energy_per_atom's metric/factor contraction.
    metric_e = coeff_e.dot(metric_e)
    dm_oo = contract('uv,nvij->nuij', metric_e, j3c_oo)
    low = j2c = raw_rho = j3c_oo = metric_e = None
    # Derivative kernels consume contiguous component rows through pointers.
    rho = auxmol.mat_dot_CT(rho_orig.T)
    charges = {t: component.charge for t, component in mf.components.items()}
    rho_dict = dict(zip(keys, rho))
    auxvec = neo_df._get_rhoj(rho[0], {t: rho_dict[t] for t in keys[1:]},
                              charges, components, with_df.df_nn)
    t0 = log.timer_debug1('NEO shared J/K metric solve', *t0)

    # Begin df.grad.rhf/uhf._jk_energy_per_atom packed derivative contractions.
    # NEO delta: add global-J weights to K's packed electronic tensor. Nuclear
    # AO pairs use the existing local-DM J kernel on a disjoint work list.
    nsp, gout_stride, shm_size = int3c2e_scheme(
        gout_width=54, deriv=(1,0,0))
    lmax = mol.uniq_l_ctr[:,0].max()
    laux = auxmol.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
    # Follow df.grad.rhf._jk_energy_per_atom's pair limits.
    pair_batch_size = np.minimum(nsp[0]*4, np.maximum(nsp[0]//4, 16))
    bas_ij_idx, shl_pair_offsets = int3c2e_bdiv._aggregate_shl_pair_blocks(
        mol, intopt.bas_ij_blocks, pair_batch_size)
    # The electronic pair prefix uses the conventional DFT derivative kernel.
    nblocks_e = 0
    for ((i, j), pairs), t in zip(intopt.bas_ij_blocks, intopt.block_components):
        if t != 'e':
            break
        li, lj = mol.uniq_l_ctr[[i, j],0]
        block_size = int(pair_batch_size[li, lj])
        nblocks_e += (len(pairs) + block_size - 1) // block_size
    ao_pair_loc = int3c2e_bdiv._get_ao_pair_loc_blocks(mol.uniq_l_ctr[:,0], intopt.bas_ij_blocks)
    offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
    offsets, _ = _split_l_ctr_pattern(offsets, auxmol.uniq_l_ctr, batch_size)
    # Retain homogeneous group offsets inside each packed derivative launch.
    aux_offsets = auxmol.ao_loc[offsets]
    ksh_offsets = cp.asarray(offsets+mol.nbas, dtype=np.int32)
    gout_stride = cp.asarray(gout_stride, dtype=np.int32)
    pair_component = cp.asarray(np.hstack([
        np.full(len(pairs), keys.index(t), dtype=np.int32)
        for (_, pairs), t in zip(intopt.bas_ij_blocks, intopt.block_components)]))
    local_ao_loc = cp.asarray(local_ao_loc, dtype=np.int32)
    component_nao = cp.asarray([component_mols[t].nao for t in keys], dtype=np.int32)
    for t in keys[1:]:
        dms[t] *= charges[t]
    auxvec_ptrs = cp.asarray(np.asarray([auxvec[t].data.ptr for t in keys], dtype=np.uintp))
    dm_ptrs = cp.asarray(np.asarray([dms[t].data.ptr for t in keys], dtype=np.uintp))
    auxvec_e = cp.empty(naux)
    auxvec_e[aux_sorting] = auxvec['e']
    packed_dm_e = dms['e'].ravel()[pair_addresses_e]
    buf = cp.empty((npair_e, batch_size))
    buf1 = cp.empty((blksize, nao, nao))
    buf2 = cp.empty((spin_sets, blksize, nocc, nao))
    ejk = cp.zeros((mol.natm, 3))
    ejk_aux = cp.zeros_like(ejk) if mf_grad.auxbasis_response else None
    aux_ptr = lib.c_null_ptr() if ejk_aux is None else ctypes.cast(ejk_aux.data.ptr, ctypes.c_void_p)
    envs = intopt.int3c2e_envs
    # Pack auxiliary groups as in df.grad.rhf._jk_energy_per_atom.
    aux_splits = splits_by_blocksize(aux_offsets, batch_size)
    for b0, b1 in zip(aux_splits[:-1], aux_splits[1:]):
        a0, a1 = aux_offsets[[b0, b1]]
        dk = a1-a0
        compressed = ndarray((npair_e, dk), buffer=buf)
        cp.multiply(packed_dm_e[:,None], auxvec_e[a0:a1], out=compressed)
        # Nuclear auxiliary gaps carry J alone; electronic rows also carry K.
        for c0, c1 in e_ranges:
            c0, c1 = max(c0, a0), min(c1, a1)
            if c0 >= c1:
                continue
            e0 = np.searchsorted(e_eval, c0)
            for p0, p1 in lib.prange(0, c1-c0, blksize):
                count = p1-p0
                dm_tensor = ndarray((nao, nao, count), buffer=buf1)
                # Vanilla J/K payload: initialize J, accumulate K, then gather.
                cp.multiply(dms['e'][:,:,None], auxvec_e[c0+p0:c0+p1], out=dm_tensor)
                if unrestricted:
                    tmp = ndarray((2, nocc, nao, count), buffer=buf2)
                    contract('nrji,nqj->niqr', dm_oo[:,e0+p0:e0+p1], dm_factor_l, out=tmp)
                    contract('niqr,npi->pqr', tmp, dm_factor_r, -k_factor, beta=1, out=dm_tensor)
                else:
                    tmp = ndarray((nocc, nao, count), buffer=buf2)
                    contract('rji,qj->iqr', dm_oo[0,e0+p0:e0+p1], dm_factor_l, out=tmp)
                    contract('iqr,pi->pqr', tmp, dm_factor_r, -.5*k_factor, beta=1, out=dm_tensor)
                cp.take(dm_tensor.reshape(-1, count), pair_addresses_e, axis=0,
                        out=compressed[:,c0-a0+p0:c0-a0+p1])
        err = df_rhf_grad.libvhf_rys.sum_ejk_int3c2e_ip1(
            ctypes.cast(ejk.data.ptr, ctypes.c_void_p), aux_ptr,
            ctypes.cast(compressed.data.ptr, ctypes.c_void_p),
            lib.c_null_ptr(), ctypes.c_int(1),
            ctypes.c_double(0), ctypes.c_double(1), ctypes.c_double(1),
            ctypes.byref(envs), ctypes.c_int(shm_size_max),
            ctypes.c_int(nblocks_e), ctypes.c_int(b1-b0),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_offsets[b0:].data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(auxmol.ao_loc[offsets[b0]]), ctypes.c_int(mol.nao),
            ctypes.c_int(npair_e), ctypes.c_int(dk), ctypes.c_int(mol.natm))
        if err != 0:
            raise RuntimeError('sum_ejk_int3c2e_ip1 failed')
    # Nuclear AO pairs use the component-local J kernel.
    if nblocks_e < len(shl_pair_offsets)-1:
        err = df_rhf_grad.libvhf_rys.sum_j_int3c2e_ip1_multi_in(
            ctypes.cast(ejk.data.ptr, ctypes.c_void_p), aux_ptr,
            ctypes.cast(dm_ptrs.data.ptr, ctypes.c_void_p),
            ctypes.cast(auxvec_ptrs.data.ptr, ctypes.c_void_p),
            ctypes.byref(envs), ctypes.c_int(shm_size_max),
            ctypes.c_int(len(shl_pair_offsets)-1-nblocks_e), ctypes.c_int(len(offsets)-1),
            ctypes.cast(shl_pair_offsets[nblocks_e:].data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(pair_component.data.ptr, ctypes.c_void_p),
            ctypes.cast(local_ao_loc.data.ptr, ctypes.c_void_p),
            ctypes.cast(component_nao.data.ptr, ctypes.c_void_p),
            ctypes.c_int(mol.natm))
        if err != 0:
            raise RuntimeError('sum_j_int3c2e_ip1_multi_in failed')
    # End adapted conventional DFT packed derivative contraction block.
    de = ejk.reshape(len(keys), natm, 3).sum(axis=0) * 2
    t0 = log.timer_debug1('NEO shared J/K int3c2e derivative', *t0)

    if mf_grad.auxbasis_response:
        # Adapt df.grad.rhf/uhf._jk_energy_per_atom's metric derivative.
        # Add electronic K to global J's source products and self-exclusions.
        charge = cp.asarray([charges[t] for t in keys])
        rho_total = charge.dot(rho)
        dm_aux = rho_total[:,None] * rho_total
        rho_n = rho[1:] * charge[1:,None]
        if with_df.df_nn:
            dm_aux = contract('ir,is->rs', rho_n, rho_n, alpha=-1, beta=1, out=dm_aux)
        else:
            rho_n = rho_n.sum(axis=0)
            dm_aux -= rho_n[:,None] * rho_n
        # Reorder the compact electronic rows back from evaluator order.
        e_cart_eval = cp.asarray(np.argsort(aux_sorting_cpu)[e_eval])
        if unrestricted:
            dm_aux_e = contract('nrij,nsij->rs', dm_oo, dm_oo, alpha=-k_factor)
        else:
            dm_aux_e = contract('rij,sij->rs', dm_oo[0], dm_oo[0], alpha=-.5*k_factor)
        dm_aux[e_cart_eval[:,None],e_cart_eval] += dm_aux_e
        de += ejk_aux[:natm] * 2
        with lib.temporary_env(auxmol, uniq_l_ctr=aux_l_ctr[0], l_ctr_counts=aux_l_ctr[1]):
            de -= cp.asarray(int2c2e_ip1_per_atom(auxmol, dm_aux))
    t0 = log.timer_debug1('NEO shared J/K int2c2e derivative', *t0)
    return de.get()


def grad_int(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None,
             k_factor=0):
    '''Evaluate global J and the explicitly requested full-range K derivative.'''
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
    if k_factor:
        de = _jk_energy_per_atom(mf_grad, dm, k_factor, 1, mf_grad.verbose)
    else:
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

    def grad_elec(self, mo_energy, mo_coeff, mo_occ, atmlst=None):
        comp = self.components['e']
        with_df = self.base.with_df
        k_factor = 0
        mf_e = comp.base
        share_ints = (not mf_e.only_dfj and
                      comp.auxbasis_response == self.auxbasis_response and
                      mf_e.with_df.auxbasis == with_df.auxbasis)
        if share_ints:
            k_factor = 1
            if hasattr(mf_e, '_numint'):
                omega, alpha, k_factor = mf_e._numint.rsh_and_hybrid_coeff(
                    mf_e.xc, spin=mf_e.mol.spin)
                if omega and mf_e.range_separated_mode == 'mix_inside_kernel':
                    k_factor = 0

        # Assign full-range K to either the joint builder or the electronic worker.
        electronic_grad = lib.set_class(
            _ElectronicGradWithoutJ(comp, with_full_range_k=not bool(k_factor)),
            (_ElectronicGradWithoutJ, comp.__class__))
        de = self.grad_int(mo_energy, mo_coeff, mo_occ, k_factor=k_factor)
        return super().grad_elec(mo_energy, mo_coeff, mo_occ, atmlst,
                                 electronic_grad=electronic_grad, interaction_grad=de)


Grad = Gradients
