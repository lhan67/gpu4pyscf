'''
NEO 3-center 2-electron Coulomb integral helper functions
'''

import ctypes
import math
import numpy as np
import cupy as cp
from pyscf import gto
from pyscf.gto.mole import ANG_OF, ATOM_OF, PTR_EXP, conc_env
from gpu4pyscf.df import int3c2e_bdiv
from gpu4pyscf.gto.mole import (
    PTR_BAS_COORD, SortedMole, RysIntEnvVars, extract_pgto_params)
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import asarray, ndarray, transpose_sum
from gpu4pyscf.lib.utils import splits_by_blocksize
from gpu4pyscf.scf.jk import (_nearest_power2, _scale_sp_ctr_coeff, _check_rsh_factors,
                              Q_COND_MARGIN, libvhf_rys)
from gpu4pyscf.df.int3c2e_bdiv import (int3c2e_scheme, _split_l_ctr_pattern,
                                       argsort_aux, _conc_locs)
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.neo import int1e

THREADS = int3c2e_bdiv.THREADS
POOL_SIZE = int3c2e_bdiv.POOL_SIZE
LMAX = int3c2e_bdiv.LMAX


def _cache_q_cond_and_non0pairs(mol, rys_envs, precision,
                                shell_component, tile=1, tril=True):
    # Copied from scf.jk._cache_q_cond_and_non0pairs.
    # NEO: shell_component identifies the owner of each sorted shell.
    from gpu4pyscf.pbc.scf.rsjk import _group_by_split_points
    assert isinstance(mol, SortedMole)
    omega = mol.omega
    ls = np.arange(LMAX+1)
    li = ls[:,None]
    lj = ls
    lij = li + lj
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    nroots = lij + 1
    if omega < 0:
        nroots *= 2

    SIZEOF_FLOAT = ctypes.sizeof(ctypes.c_float)
    gout_width = 29
    unit = (li+1)*(lj+1)*2 + (li+1)*(lj+1)*(lij+1) + 6 + nroots*2
    shm_size = 1024 * 48 - 1024
    nsp_max = _nearest_power2(shm_size // (unit*SIZEOF_FLOAT))
    gout_size = nfi * nfj
    gout_stride = (gout_size+gout_width-1) // gout_width
    gout_stride = _nearest_power2(gout_stride, return_leq=False)
    nsp_per_block = THREADS // gout_stride
    # min(nsp_per_block, nsp_max)
    nsp_per_block = np.where(nsp_per_block < nsp_max, nsp_per_block, nsp_max)
    gout_stride = THREADS // nsp_per_block
    gout_stride = cp.asarray(gout_stride, dtype=np.int32)
    shm_size = nsp_per_block * (unit*SIZEOF_FLOAT)
    # (pp|pp) requires more shm than this estimation. 5888 is the required size
    max_shm_size = max(shm_size.max(), 5888*SIZEOF_FLOAT)

    # NEO: screen compact component-local candidates in the original tile order.
    # Neither overlap flags nor the pair buffer span the concatenated square.
    screened_pairs = int1e._generate_shl_pairs(
        mol, shell_component, hermi=int(tril), precision=precision**2, tile=tile)
    uniq_l = mol.uniq_l_ctr[:,0]
    n_groups = np.count_nonzero(uniq_l <= LMAX)
    if tril:
        pair_keys = ((i, j) for i in range(n_groups) for j in range(i+1))
    else:
        pair_keys = ((i, j) for i in range(n_groups) for j in range(n_groups))
    bas_ij_cache = {} # The effective shell pair = ish*nbas+jsh
    shl_pair_offsets = [] # the bas_ij_idx offset for each blockIdx.x
    sp0 = sp1 = 0
    for i, j in pair_keys:
        li = uniq_l[i]
        lj = uniq_l[j]
        pair_ij = screened_pairs[i,j]
        bas_ij_cache[i,j] = cp.asarray(pair_ij, dtype=np.uint32)
        nshl_pair = len(pair_ij)
        sp0, sp1 = sp1, sp1 + nshl_pair
        nsp_per_block = THREADS // gout_stride[li, lj] * 8
        shl_pair_offsets.append(np.arange(sp0, sp1, nsp_per_block, dtype=np.int32))
    screened_pairs = None
    shl_pair_offsets.append(np.int32(sp1))
    shl_pair_offsets = cp.array(np.hstack(shl_pair_offsets), dtype=np.int32)
    bas_ij_counts = [len(x) for x in bas_ij_cache.values()]
    bas_ij_cum = np.cumsum(bas_ij_counts)
    bas_ij_idx = cp.array(cp.hstack(bas_ij_cache.values()), dtype=np.uint32)

    lr_factor = sr_factor = 1
    if omega < 0:
        lr_factor = 0
    if omega > 0:
        sr_factor = 0
    nbatches_shl_pair = len(shl_pair_offsets) - 1
    q_out = cp.empty(len(bas_ij_idx), dtype=np.float32)
    libvhf_rys.int2e_qcond_estimator.restype = ctypes.c_int
    err = libvhf_rys.int2e_qcond_estimator(
        ctypes.cast(q_out.data.ptr, ctypes.c_void_p),
        ctypes.byref(rys_envs),
        ctypes.c_int(max_shm_size),
        ctypes.c_int(nbatches_shl_pair),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
        ctypes.c_double(omega),
        ctypes.c_double(lr_factor),
        ctypes.c_double(sr_factor))
    if err != 0:
        raise RuntimeError('int2e_qcond_estimator kernel failed')

    if omega < 0:
        diffuse_exps, diffuse_ctr_coef = extract_pgto_params(mol, 'diffuse')
        diffuse_exps = cp.asarray(diffuse_exps, dtype=np.float32)
        diffuse_ctr_coef = cp.asarray(diffuse_ctr_coef, dtype=np.float32)
        s_out = cp.empty(len(bas_ij_idx), dtype=np.float32)
        libvhf_rys.fill_s_estimator.restype = ctypes.c_int
        err = libvhf_rys.fill_s_estimator(
            ctypes.cast(s_out.data.ptr, ctypes.c_void_p),
            ctypes.byref(rys_envs),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_ctr_coef.data.ptr, ctypes.c_void_p),
            ctypes.c_int(len(bas_ij_idx)),
            ctypes.c_double(omega))
        if err != 0:
            raise RuntimeError('fill_s_estimator kernel failed')

    split_points = cp.arange(math.log(precision), 2., Q_COND_MARGIN)
    q_cond_cache = {}
    q_cond = cp.split(q_out, bas_ij_cum[:-1])
    if omega < 0:
        s_cond = cp.split(s_out, bas_ij_cum[:-1])
    for i, key in enumerate(bas_ij_cache):
        idx = _group_by_split_points(q_cond[i], split_points)
        pair_ij = bas_ij_cache[key][idx]
        q_cond_ij = q_cond[i][idx]
        s_cond_ij = q_cond_ij
        if omega < 0:
            s_cond_ij = s_cond[i][idx]
        q_cond_cache[key] = pair_ij, q_cond_ij, s_cond_ij
    # End copied block.
    return q_cond_cache


def _aggregate_shl_pair_blocks(mol, bas_ij_blocks, nsp_per_block=512):
    # bas_ij_blocks contains ((l_i,l_j), bas_i*nbas+bas_j) entries in
    # component order. bas_ij_idx concatenates those encoded shell pairs;
    # shl_pair_offsets partitions them into CUDA work blocks without merging
    # equal angular keys belonging to different components.
    bas_ij_idx = []
    shl_pair_offsets = []
    sp0 = sp1 = 0
    l = mol.uniq_l_ctr[:,0]
    for (i, j), bas_ij in bas_ij_blocks:
        bas_ij_idx.append(cp.asarray(bas_ij))
        sp0, sp1 = sp1, sp1 + len(bas_ij)
        if isinstance(nsp_per_block, (int, np.integer)):
            batch_size = nsp_per_block
        else:
            batch_size = nsp_per_block[l[i], l[j]]
        shl_pair_offsets.append(cp.arange(sp0, sp1, batch_size, dtype=np.int32))
    bas_ij_idx = cp.asarray(cp.hstack(bas_ij_idx), dtype=np.int32)
    shl_pair_offsets.append(np.int32(sp1))
    shl_pair_offsets = cp.asarray(cp.hstack(shl_pair_offsets), dtype=np.int32)
    return bas_ij_idx, shl_pair_offsets


def _get_ao_pair_loc_blocks(uniq_l, bas_ij_blocks, cart=True):
    '''
    For each primitive shell-pair in bas_ij_idx, ao_pair_loc points to the
    addresses of first element for the contracted pair-GTOs. In each
    shell-pair, there are nfij elements. Note, the nfij elements are
    sorted as [nfj,nfi] (in F-order).

    Repeated angular-pair blocks are retained because each block belongs to a
    distinct component.
    '''
    if cart:
        nf = (uniq_l + 1) * (uniq_l + 2) // 2
    else:
        nf = uniq_l * 2 + 1
    ao_pair_loc = []
    p0 = p1 = 0
    for (i, j), bas_ij in bas_ij_blocks:
        nfij = nf[i] * nf[j]
        p0, p1 = p1, p1 + nfij * len(bas_ij)
        ao_pair_loc.append(cp.arange(p0, p1, nfij, dtype=np.int32))
    ao_pair_loc.append(np.int32(p1))
    ao_pair_loc = cp.hstack(ao_pair_loc, dtype=np.int32)
    return ao_pair_loc


def _create_pair_recontractor(intopt, int3c2e_context):
    # pair_addresses initially index the square AO matrix of the concatenated
    # original molecule. cderi_idx[t] converts the addresses belonging to t
    # into flattened indices of the component-local (nao_t,nao_t) matrix.
    recontract, ao_pair_counts, contracted_ao_pair_counts, pair_addresses = \
            int3c2e_bdiv._create_pair_recontractor(intopt.mol, int3c2e_context)

    cderi_idx = {}
    nao = int(intopt.mol.mol.ao_loc[-1])
    rows = pair_addresses // nao
    for t in intopt.component_names:
        ao0 = intopt.component_ao_offsets[t]
        ao1 = ao0 + intopt.component_nao[t]
        mask = (ao0 <= rows) & (rows < ao1)
        pair_addresses_t = pair_addresses[mask]
        rows_t, cols_t = divmod(pair_addresses_t, nao)
        nao_t = intopt.component_nao[t]
        pair_addresses_t = (rows_t - ao0) * nao_t + cols_t - ao0
        diag_t = np.where(rows_t == cols_t)[0].astype(np.int32)
        cderi_idx[t] = asarray(pair_addresses_t), asarray(diag_t)
    return recontract, ao_pair_counts, contracted_ao_pair_counts, cderi_idx


class Int3c2eOpt(int3c2e_bdiv.Int3c2eOpt):
    def __init__(self, components, auxmol):
        # components maps labels ('e', 'n1', ...) to independent Mole objects.
        # The concatenated mol supplies shared integral arrays; the mappings
        # below recover each component's atom and AO ranges.
        self.components = components
        component_names = list(components)
        mols = [components[t] for t in component_names]
        atom_component = []
        component_ao_offsets = []
        component_nao = {}
        mol = mols[0]
        atom_component.extend([0] * mol.natm)
        ao0 = 0
        for t, mol_t in zip(component_names, mols):
            component_ao_offsets.append(ao0)
            component_nao[t] = mol_t.nao
            ao0 += mol_t.nao
        for ic, mol1 in enumerate(mols[1:], 1):
            mol = gto.conc_mol(mol, mol1)
            atom_component.extend([ic] * mol1.natm)
        super().__init__(mol, auxmol)
        self.component_names = component_names
        self.component_nao = component_nao
        self.component_ao_offsets = dict(zip(component_names, component_ao_offsets))
        self.atom_component = np.asarray(atom_component, dtype=np.int32)
        # bas_ij_blocks stores ((l_i,l_j), encoded shell-pair array), and the
        # parallel block_components entry identifies the component owning each
        # block. Their order defines grouped integral rows and CDERI columns.
        self.bas_ij_blocks = None
        self.block_components = None

    def build(self, cutoff=1e-14, tril=True):
        # Copied from df.int3c2e_bdiv.Int3c2eOpt.build.
        mol = self.mol = SortedMole.from_cell(self.mol)
        auxmol = self.auxmol = SortedMole.from_cell(self.auxmol)
        _atm, _bas, _env = conc_env(
            mol._atm, mol._bas, _scale_sp_ctr_coeff(mol),
            auxmol._atm, auxmol._bas, _scale_sp_ctr_coeff(auxmol))
        #NOTE: PTR_BAS_COORD is not updated in conc_env()
        off = _bas[mol.nbas,PTR_EXP] - auxmol._bas[0,PTR_EXP]
        _bas[mol.nbas:,PTR_BAS_COORD] += off
        ao_loc = mol.ao_loc
        aux_loc = auxmol.ao_loc
        ao_loc = cp.asarray(_conc_locs(ao_loc, aux_loc), dtype=np.int32)
        self._int3c2e_envs = RysIntEnvVars.new(
            mol.natm, mol.nbas, _atm, _bas, _env, ao_loc)

        # NEO: remove cross-component shell pairs before overlap and
        # q-condition screening constructs the AO-pair work list.
        shell_component = self.atom_component[mol._bas[:,ATOM_OF]]
        bas_pair_cache = _cache_q_cond_and_non0pairs(
            mol, self._int3c2e_envs, cutoff, shell_component,
            tile=4, tril=tril)
        self.bas_ij_cache = {
            k: cp.sort(v[0]) for k, v in mol.iter_pair_by_l(bas_pair_cache)}
        # End copied block.

        component_names = self.component_names

        nbas = mol.nbas
        l_ctr_offsets = np.append(0, np.cumsum(mol.l_ctr_counts))
        component_ids = np.arange(len(component_names) + 1)
        component_blocks = [[] for _ in component_names]
        pair_blocks = []
        block_offsets = []
        for k, pair_ij in self.bas_ij_cache.items():
            if pair_ij.size == 0:
                continue
            # SortedMole keeps component shells contiguous within each angular
            # group, so the parent's encoded-pair order groups components.
            pair_blocks.append((k, pair_ij))
            i = k[0]
            ish0, ish1 = l_ctr_offsets[i:i+2]
            shell_starts = ish0 + np.searchsorted(shell_component[ish0:ish1], component_ids)
            pair_starts = cp.asarray(shell_starts*nbas, dtype=pair_ij.dtype)
            block_offsets.append(cp.searchsorted(pair_ij, pair_starts))
        block_offsets = cp.stack(block_offsets).get()
        for (k, pair_ij), offsets in zip(pair_blocks, block_offsets):
            for ic, (p0, p1) in enumerate(zip(offsets[:-1], offsets[1:])):
                if p0 != p1:
                    component_blocks[ic].append((k, pair_ij[p0:p1]))

        # Append angular blocks in component order so each component occupies
        # one contiguous range of grouped integral rows and CDERI columns.
        bas_ij_blocks = []
        block_components = []
        for t, blocks in zip(component_names, component_blocks):
            bas_ij_blocks.extend(blocks)
            block_components.extend([t] * len(blocks))
        self.bas_ij_blocks = bas_ij_blocks
        self.block_components = block_components
        return self

    def _component_mol_data(self):
        # component_mols[t] is the independently sorted Mole used
        # to transform dm[t]. The combined optimizer and each component sort
        # their shells independently, so their sorted AO indices differ.
        mol = self.mol
        component_mols = {
            t: SortedMole.from_cell(self.components[t])
            for t in self.component_names}
        # shell_local[ish] is the original component-local shell number for
        # sorted combined shell ish. local_ao_loc[ish] then gives the first AO
        # of that shell in component_mols[t].
        shell_component = self.atom_component[mol._bas[:,ATOM_OF]]
        shell_local = np.concatenate(
            [np.arange(component_mols[t].nbas) for t in self.component_names]
        )[mol.sorted_idx]
        local_ao_loc = np.empty(mol.nbas, dtype=np.int32)
        for ic, t in enumerate(self.component_names):
            mol_t = component_mols[t]
            inv_sorted = np.empty_like(mol_t.sorted_idx)
            inv_sorted[mol_t.sorted_idx] = np.arange(mol_t.nbas)
            idx = np.where(shell_component == ic)[0]
            local_ao_loc[idx] = mol_t.ao_loc[inv_sorted[shell_local[idx]]]

        return component_mols, local_ao_loc

    def int3c2e_evaluator(self, ao_pair_batch_size=None, aux_batch_size=None,
                          reorder_aux=False, cart=None, pair_batch_by_l=False,
                          return_clone_context=False, clone_context=None,
                          omega=None, lr_factor=None, sr_factor=None):
        if self._int3c2e_envs is None:
            self.build()
        mol = self.mol
        auxmol = self.auxmol
        omega, lr_factor, sr_factor = _check_rsh_factors(mol, omega, lr_factor, sr_factor)

        nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
            short_range=omega<0, gout_width=54, cache_cart_idx=True)
        gout_stride = cp.asarray(gout_stride, dtype=np.int32)
        lmax = mol.uniq_l_ctr[:,0].max()
        laux = auxmol.uniq_l_ctr[:,0].max()
        shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()

        if clone_context is None:
            if cart is None:
                cart = mol.mol.cart

            # bas_ij_idx lists encoded shell pairs in component-block order.
            # ao_pair_loc[p] is the first flattened AO-pair row generated for
            # bas_ij_idx[p], so its final value is sum_t npair_t.
            bas_ij_blocks = self.bas_ij_blocks
            bas_ij_idx, shl_pair_offsets = _aggregate_shl_pair_blocks(
                mol, bas_ij_blocks, nsp_per_block[0]*4)
            ao_pair_loc = _get_ao_pair_loc_blocks(mol.uniq_l_ctr[:,0], bas_ij_blocks, cart)
            nao_pair = ao_pair_loc[-1].get()

            if ao_pair_batch_size is None or nao_pair <= ao_pair_batch_size:
                pair_splits = np.array([0, len(shl_pair_offsets)-1])
                ao_pair_offsets = np.array([0, nao_pair])

            elif pair_batch_by_l:
                uniq_l = mol.uniq_l_ctr[:,0].tolist()
                pair_batch_sizes = []
                last_key = None
                for (i, j), bas_ij in bas_ij_blocks:
                    key = (uniq_l[i], uniq_l[j])
                    if key != last_key:
                        last_key = key
                        pair_batch_sizes.append(bas_ij.size)
                    else:
                        pair_batch_sizes[-1] += bas_ij.size
                ao_pair_offsets = ao_pair_loc[shl_pair_offsets].get()
                pair_splits = np.append(0, np.searchsorted(
                    shl_pair_offsets.get(), np.cumsum(pair_batch_sizes), side='left'))
                ao_pair_offsets = ao_pair_offsets[pair_splits]
                assert max(ao_pair_offsets[1:]-ao_pair_offsets[:-1]) < ao_pair_batch_size

            else:
                ao_pair_offsets = ao_pair_loc[shl_pair_offsets].get()
                pair_splits = splits_by_blocksize(ao_pair_offsets, ao_pair_batch_size)
                ao_pair_offsets = ao_pair_offsets[pair_splits]

            l_ctr_aux_offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
            uniq_l_ctr_aux = auxmol.uniq_l_ctr
            aux_loc = auxmol.ao_loc
            if aux_batch_size is None:
                ksh_offsets_cpu = l_ctr_aux_offsets
                aux_splits = [0, len(ksh_offsets_cpu)-1]
            else:
                l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
                    l_ctr_aux_offsets, uniq_l_ctr_aux, aux_batch_size)
                ksh_offsets_cpu = l_ctr_aux_offsets
                aux_splits = range(len(ksh_offsets_cpu))
            aux_offsets = aux_loc[ksh_offsets_cpu[aux_splits]]
            if reorder_aux:
                aux_sorting = argsort_aux(l_ctr_aux_offsets, uniq_l_ctr_aux)
            else:
                aux_sorting = slice(aux_loc[-1])

            # Save the context for rebuilding evaluate_j3c on multiple GPUs
            clone_context = (bas_ij_idx, shl_pair_offsets, pair_splits, cart,
                             ksh_offsets_cpu, aux_splits, aux_sorting)
        else:
            # Restore the context for running evaluate_j3c on multiple GPUs
            bas_ij_idx, shl_pair_offsets, pair_splits, cart, \
                    ksh_offsets_cpu, aux_splits, aux_sorting = clone_context

            bas_ij_idx = cp.asarray(bas_ij_idx)
            shl_pair_offsets = cp.asarray(shl_pair_offsets)

            ish, jsh = divmod(bas_ij_idx, mol.nbas)
            ls = mol._bas[:,ANG_OF]
            if cart:
                nf = cp.asarray((ls + 1) * (ls + 2) // 2)
            else:
                nf = cp.asarray(ls * 2 + 1)
            ao_pair_counts = nf[ish] * nf[jsh]
            ao_pair_loc = cp.asarray(cp.append(np.int32(0), ao_pair_counts.cumsum()), dtype=np.int32)
            ao_pair_offsets = ao_pair_loc[shl_pair_offsets[pair_splits]].get()

            aux_loc = auxmol.ao_loc
            aux_offsets = aux_loc[ksh_offsets_cpu[aux_splits]]
            if isinstance(aux_sorting, (np.ndarray, cp.ndarray)):
                # aux_sorting can be a slice() instance
                aux_sorting = cp.asarray(aux_sorting)

        ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)
        shl_pair_batches = len(ao_pair_offsets) - 1
        aux_batches = len(aux_offsets) - 1
        logger.debug1(self.mol, 'sp_batches = %d, ksh_batches = %d',
                      shl_pair_batches, aux_batches)

        workers = gpu_specs['multiProcessorCount']
        pool = cp.empty(workers * POOL_SIZE + 1)
        kern = libvhf_rys.fill_int3c2e
        int3c2e_envs = self.int3c2e_envs

        def evaluate_j3c(shl_pair_batch_id=0, aux_batch_id=0, out=None):
            pair_split0 = pair_splits[shl_pair_batch_id]
            pair_split1 = pair_splits[shl_pair_batch_id+1]
            ao_pair_offset = ao_pair_offsets[shl_pair_batch_id]
            nao_pair = ao_pair_offsets[shl_pair_batch_id+1] - ao_pair_offset

            aux_split0 = aux_splits[aux_batch_id]
            aux_split1 = aux_splits[aux_batch_id+1]
            ksh0 = ksh_offsets_cpu[aux_split0]
            ksh1 = ksh_offsets_cpu[aux_split1]
            aux_ao_offset = aux_loc[ksh0]
            naux = aux_loc[ksh1] - aux_ao_offset
            out = ndarray((nao_pair, naux), buffer=out)
            if not cart:
                out[:] = 0.
            if out.size == 0:
                return out
            err = kern(
                ctypes.cast(out.data.ptr, ctypes.c_void_p),
                ctypes.byref(int3c2e_envs),
                ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                ctypes.c_double(omega),
                ctypes.c_double(lr_factor), ctypes.c_double(sr_factor),
                ctypes.c_int(shm_size_max),
                ctypes.c_int(pair_split1 - pair_split0),
                ctypes.c_int(aux_split1 - aux_split0),
                ctypes.cast(shl_pair_offsets[pair_split0:].data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(ksh_offsets_gpu[aux_split0:].data.ptr, ctypes.c_void_p),
                ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
                ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
                ctypes.c_int(ao_pair_offset), ctypes.c_int(aux_ao_offset),
                ctypes.c_int(naux), ctypes.c_int(reorder_aux),
                ctypes.c_int(not cart))
            if err != 0:
                raise RuntimeError('fill_int3c2e kernel failed')
            return out


        if return_clone_context:
            return (evaluate_j3c, aux_sorting, ao_pair_offsets, aux_offsets,
                    clone_context)
        else:
            return evaluate_j3c, aux_sorting, ao_pair_offsets, aux_offsets

    def pair_and_diag_indices(self, cart=None, original_ao_order=True):
        # ao_pair_addresses[t] lists flattened AO-matrix addresses in the same
        # sequence as component t's grouped integral rows. diag[t] contains all
        # AO-pair rows from shell pairs with ish == jsh, not only AO diagonals.
        mol = self.mol
        if cart is None:
            cart = mol.mol.cart
        nbas = mol.nbas
        ao_loc = mol.ao_loc_nr(cart=cart)
        nao = ao_loc[-1]
        ao_loc = cp.asarray(ao_loc)
        uniq_l = mol.uniq_l_ctr[:,0]
        if cart:
            nf = (uniq_l + 1) * (uniq_l + 2) // 2
        else:
            nf = uniq_l * 2 + 1
        carts = [cp.arange(n) for n in nf]
        ao_idx_mapping = None
        if original_ao_order:
            # ao_idx_mapping converts sorted combined AO indices to the
            # concatenated original AO order. Subtracting component_ao_offsets
            # below then produces flattened addresses in each original DM.
            ao_idx_mapping = np.empty(nao, dtype=int)
            ao_idx_mapping[mol.get_ao_idx(cart)] = np.arange(nao)
            ao_idx_mapping = cp.array(ao_idx_mapping)

        offset = dict.fromkeys(self.component_names, 0)
        diag = {t: [] for t in self.component_names}
        ao_pair_addresses = {t: [] for t in self.component_names}
        for ((i, j), bas_ij), t in zip(self.bas_ij_blocks, self.block_components):
            ish, jsh = divmod(bas_ij, nbas)
            iaddr = ao_loc[ish,None] + carts[i]
            jaddr = ao_loc[jsh,None] + carts[j]
            addr = (iaddr[:,None,:] * nao + jaddr[:,:,None]).ravel()
            if original_ao_order:
                row, col = divmod(addr, nao)
                addr = ao_idx_mapping[row] * nao + ao_idx_mapping[col]
                row, col = divmod(addr, nao)
                ao0 = self.component_ao_offsets[t]
                nao_t = self.component_nao[t]
                addr = (row - ao0) * nao_t + col - ao0
            ao_pair_addresses[t].append(addr)
            if i == j: # the diagonal blocks
                nfi = nf[i]
                idx = cp.where(ish == jsh)[0]
                addr = offset[t] + idx[:,None] * (nfi*nfi) + cp.arange(nfi*nfi)
                diag[t].append(addr.ravel())
            offset[t] += len(bas_ij) * nf[i] * nf[j]

        out = {}
        for t in self.component_names:
            ao_pair_address = cp.hstack(ao_pair_addresses[t])
            if diag[t]:
                diag_t = cp.hstack(diag[t])
            else:
                diag_t = cp.asarray([], dtype=np.int32)
            out[t] = ao_pair_address, diag_t
        return out

    def contract_dm(self, dm, hermi=0, return_transformed_dm=False):
        if self._int3c2e_envs is None:
            self.build()
        log = logger.new_logger(self.mol)
        t0 = log.init_timer()
        mol = self.mol
        auxmol = self.auxmol
        component_mols, local_ao_loc = self._component_mol_data()

        # Each component DM is transformed to the independently sorted AO basis
        # addressed by local_ao_loc in the component-input contraction kernel.
        dms = {}
        dm_ndim = None
        n_dm = None
        for t in self.component_names:
            dm_t = component_mols[t].apply_C_mat_CT(cp.asarray(dm[t]))
            nao = component_mols[t].nao
            assert dm_t.dtype == np.float64
            if hermi != 1:
                dm_t = transpose_sum(dm_t, inplace=True)
            if dm_ndim is None:
                dm_ndim = dm_t.ndim
            else:
                assert dm_t.ndim == dm_ndim
            dm_t = cp.asarray(dm_t.reshape(-1, nao, nao), order='C')
            dms[t] = dm_t
            if n_dm is None:
                n_dm = len(dm_t)
            elif len(dm_t) != n_dm:
                raise ValueError('All component density matrices must have the same number of sets')

        nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
            short_range=mol.omega<0, cache_cart_idx=True)
        lmax = mol.uniq_l_ctr[:,0].max()
        laux = auxmol.uniq_l_ctr[:,0].max()
        shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
        bas_ij_idx, shl_pair_offsets = _aggregate_shl_pair_blocks(
            mol, self.bas_ij_blocks, nsp_per_block[0]*16)
        gout_stride = cp.asarray(gout_stride, dtype=np.int32)

        # Repeated angular blocks remain component-homogeneous, allowing each
        # CUDA block to select one local DM and one local auxiliary output.
        component_index = {t: i for i, t in enumerate(self.component_names)}
        pair_component = cp.asarray(np.hstack([
            np.full(len(bas_ij), component_index[t], dtype=np.int32)
            for (_, bas_ij), t in zip(self.bas_ij_blocks, self.block_components)]))
        dm_ptrs = [dms[t].data.ptr for t in self.component_names]
        dm_ptrs = cp.asarray(np.asarray(dm_ptrs, dtype=np.uintp))
        local_ao_loc = cp.asarray(local_ao_loc, dtype=np.int32)
        component_nao = [component_mols[t].nao for t in self.component_names]
        component_nao = cp.asarray(component_nao, dtype=np.int32)

        int3c2e_envs = self.int3c2e_envs
        naux = auxmol.nao
        vj_aux = cp.zeros((len(self.component_names), n_dm, naux))
        vj_ptrs = [vj_aux[i].data.ptr for i in range(len(self.component_names))]
        vj_ptrs = cp.asarray(np.asarray(vj_ptrs, dtype=np.uintp))
        err = libvhf_rys.contract_int3c2e_dm_multi_inout(
            ctypes.cast(vj_ptrs.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm_ptrs.data.ptr, ctypes.c_void_p),
            ctypes.c_int(n_dm), ctypes.c_int(naux),
            ctypes.byref(int3c2e_envs), ctypes.c_int(shm_size_max),
            ctypes.c_int(auxmol.nbas),
            ctypes.c_int(len(shl_pair_offsets) - 1),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(pair_component.data.ptr, ctypes.c_void_p),
            ctypes.cast(local_ao_loc.data.ptr, ctypes.c_void_p),
            ctypes.cast(component_nao.data.ptr, ctypes.c_void_p))
        if err != 0:
            raise RuntimeError('contract_int3c2e_dm_multi_inout failed')
        if hermi == 1:
            vj_aux *= 2

        if dm_ndim == 2:
            vj_aux = vj_aux[:,0]
        log.timer_debug1('processing contract_int3c2e_dm', *t0)
        vj_aux = {t: vj_aux[i] for i, t in enumerate(self.component_names)}
        if return_transformed_dm:
            return vj_aux, dms, local_ao_loc
        return vj_aux
