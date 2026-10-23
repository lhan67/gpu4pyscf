import ctypes
import cupy
import cupy as cp
import numpy as np
from cupyx.scipy.linalg import solve_triangular
from pyscf import lib
from pyscf.df import addons
from pyscf.neo import df as df_cpu
from gpu4pyscf import scf
from gpu4pyscf.df import df, df_jk, int3c2e_bdiv
from gpu4pyscf.df import j_engine_3c2e as df_j_engine_3c2e
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.lib.cupy_helper import (
    asarray, cholesky, contract, empty_mapped, fill_symmetric, get_avail_mem,
    ndarray, tag_array)
from gpu4pyscf.gto.mole import SortedMole
from gpu4pyscf.neo import hf, int3c2e_bdiv as neo_int3c2e_bdiv
from gpu4pyscf.neo import j_engine_3c2e, ks

libvhf_rys = int3c2e_bdiv.libvhf_rys


class DF(df.DF):
    '''Density fitting for multicomponent NEO Coulomb terms.

    This class follows gpu4pyscf.df.df.DF, but the AO side is a dict over
    quantum components while the auxiliary metric is shared.
    '''

    _keys = df.DF._keys.union({
        'df_ne_scheme', 'df_nn', 'nuc_auxbasis', 'nuc_auxbasis_beta',
        'nuc_auxbasis_lmax',
        'df_ne_component_vint', 'df_ne_j_engine',
        '_charges', '_elec_with_df', '_auxmol_atom_major'})

    def __init__(self, mol, auxbasis=None, df_ne_scheme='global',
                 nuc_auxbasis=None, nuc_auxbasis_beta=2.0,
                 nuc_auxbasis_lmax=None,
                 df_nn=True, df_ne_component_vint=False,
                 df_ne_j_engine='direct'):
        super().__init__(mol, auxbasis)
        self.df_ne_scheme = df_ne_scheme
        self.nuc_auxbasis = nuc_auxbasis
        self.nuc_auxbasis_beta = nuc_auxbasis_beta
        self.nuc_auxbasis_lmax = nuc_auxbasis_lmax
        self.df_nn = df_nn
        self.df_ne_component_vint = df_ne_component_vint
        self.df_ne_j_engine = df_ne_j_engine
        self.intopt = {}
        self.j_engine = {}
        self.nao = {}
        self._charges = {}
        self._elec_with_df = None
        self._auxmol_atom_major = None

    __getstate__, __setstate__ = lib.generate_pickle_methods(
        excludes=('intopt', '_cderi', '_auxmol_atom_major'))

    def make_auxmol(self):
        return self.to_cpu().make_auxmol()

    def make_auxmol_atom_major(self):
        return self.to_cpu().make_auxmol_atom_major()

    def build(self, *, direct_scf_tol=None,
              omega=None, lr_factor=None, sr_factor=None,
              build_cderi=True):
        if omega is not None or lr_factor is not None or sr_factor is not None:
            raise ValueError('RSH-DF is handled by the electronic component.')
        mol = self.mol
        auxmol = self.auxmol
        log = logger.new_logger(mol, mol.verbose)
        t0 = log.init_timer()
        if self.df_ne_scheme != 'global':
            raise NotImplementedError('GPU DF-NE only supports df_ne_scheme="global"')
        if auxmol is None:
            self.auxmol = auxmol = self.make_auxmol()
        self.naux = auxmol.nao

        self.nao = {t: mol_t.nao for t, mol_t in mol.components.items()}
        unsupported_components = [t for t in mol.components if t != 'e' and not t.startswith('n')]
        assert not unsupported_components, \
            f'Unsupported NEO component keys {unsupported_components}'
        nuc_keys = [t for t in mol.components if t.startswith('n')]
        if not build_cderi and self.df_ne_j_engine == 'direct':
            self.j_engine = {}
            self.j_engine['e'] = df_j_engine_3c2e.Int3c2eOpt(mol.components['e'], auxmol).build()
            if nuc_keys:
                nuc_mols = {t: mol.components[t] for t in nuc_keys}
                self.j_engine['n'] = j_engine_3c2e.Int3c2eOpt(nuc_mols, auxmol).build()
            return self

        build_e_cderi = (self.df_ne_j_engine == 'cderi' and
                         self.df_ne_scheme == 'global' and
                         getattr(self, '_build_e_cderi', False) and
                         self._elec_with_df is not None and
                         getattr(self._elec_with_df, '_cderi', None) is None)

        auxmol_e = None
        if build_e_cderi:
            auxmol_e = addons.make_auxmol(mol.components['e'], self.auxbasis)
        self.intopt = {}
        intopt = int3c2e_bdiv.Int3c2eOpt(mol.components['e'], auxmol)
        intopt.mol = SortedMole.from_mol(mol.components['e'], decontract=True)
        intopt.build()
        self.intopt['e'] = intopt
        if nuc_keys:
            intopt = neo_int3c2e_bdiv.Int3c2eOpt({t: mol.components[t] for t in nuc_keys}, auxmol)
            intopt.build()
            self.intopt['n'] = intopt

        if build_cderi:
            self._cderi, self._cderi_idx, naux_e = _cholesky_eri(
                self.intopt, auxmol_e=auxmol_e, orig_auxmol=auxmol,
                omega=omega, use_gpu_memory=self.use_gpu_memory)
            if build_e_cderi and naux_e is not None:
                self._elec_with_df.reset()
                self._elec_with_df.auxmol = auxmol_e
                self._elec_with_df.nao = self.nao['e']
                self._elec_with_df.naux = naux_e
                self._elec_with_df.intopt = self.intopt['e']
                self._elec_with_df._cderi_idx = self._cderi_idx['e']
                # Electronic-only CDERI occupies the first naux_e global rows.
                # Keep each device's share of those rows as a view.
                self._elec_with_df._cderi = []
                aux0 = 0
                for cderi in self._cderi['e']:
                    self._elec_with_df._cderi.append(cderi[:max(0, naux_e-aux0)])
                    aux0 += cderi.shape[0]
            log.timer_debug1('cholesky_eri', *t0)
        return self

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=None,
               omega=None, lr_factor=None, sr_factor=None):
        if omega is not None or lr_factor is not None or sr_factor is not None:
            raise ValueError('RSH-DF is handled by the electronic component.')
        if not with_k and self._cderi is None and self.df_ne_j_engine == 'direct':
            return get_j(self, dm, hermi), None
        return get_jk(self, dm, hermi, with_j, with_k, omega=omega,
                      lr_factor=lr_factor, sr_factor=sr_factor)

    def get_blksize(self, extra=0, nao=None, mem_fraction=0.3,
                    blksize_cap=None, unpack=True):
        '''
        extra for pre-calculated space for other variables
        '''
        if nao is None: nao = self.nao
        assert isinstance(nao, dict)
        device_id = cp.cuda.Device().id
        components = tuple(nao.keys())
        nuc_keys = [t for t in components if t.startswith('n')]
        storage_keys = ['e']
        if nuc_keys:
            storage_keys.append('n')
        cderi_sparse = {t: self._cderi[t][device_id] for t in storage_keys}
        naux_slice = cderi_sparse['e'].shape[0]
        on_gpu = {t: isinstance(cderi_sparse[t], cp.ndarray) for t in storage_keys}
        if naux_slice == 0:
            # On multiple GPUs, cderi_sparse might be a zero-sized array.
            # return a non-zero blksize to avoid potential issues in workspace
            # size estimation and offsets computation.
            return 1

        denom = extra
        if unpack:
            for t, nao_t in nao.items():
                denom += nao_t * nao_t
        if not on_gpu['e']:
            denom += len(self._cderi_idx['e'][0]) * 2
        if nuc_keys and not on_gpu['n']:
            denom += sum(len(self._cderi_idx[t][0]) for t in nuc_keys) * 2
        mem_avail = get_avail_mem()
        word_avail = int(mem_avail * mem_fraction / 8)

        if denom == 0:
            blksize = naux_slice
        else:
            blksize = word_avail // denom
        if blksize > df.ALIGNED:
            blksize = blksize // df.ALIGNED * df.ALIGNED
        if blksize_cap is not None:
            blksize = min(blksize, blksize_cap)
        blksize = min(blksize, naux_slice)
        logger.debug1(self.mol, f"{mem_avail/1e9:.3f} GB memory available on"
                      f"Device {device_id}, block size = {blksize}")
        assert blksize > 0
        return blksize

    def loop(self, blksize=None, unpack=True):
        ''' loop over cderi for the current device
            and unpack the CDERI in (Lij) format
        '''
        if self._cderi is None:
            self.build()

        device_id = cupy.cuda.Device().id
        nuc_keys = [t for t in self.mol.components if t.startswith('n')]
        storage_keys = ['e']
        if nuc_keys:
            storage_keys.append('n')
        cderi_sparse = {t: self._cderi[t][device_id] for t in storage_keys}
        nao = self.nao
        naux_slice = cderi_sparse['e'].shape[0]
        if blksize is None:
            blksize = self.get_blksize()
        blksize = min(blksize, naux_slice)
        if blksize == 0:
            # On multiple GPUs, cderi_sparse might be a zero-sized array
            return

        on_gpu = {t: isinstance(cderi_sparse_t, cp.ndarray)
                  for t, cderi_sparse_t in cderi_sparse.items()}
        for t, cderi_sparse_t in cderi_sparse.items():
            if cderi_sparse_t.shape[0] != naux_slice:
                raise RuntimeError(f'Inconsistent CDERI auxiliary dimension for {t}')

        if unpack:
            work = {t: cp.zeros((nao[t], nao[t], blksize)) for t in self.mol.components}

        pair_idx = {t: cp.asarray(self._cderi_idx[t][0], dtype=np.int32)
                    for t in self.mol.components}
        npairs = {'e': len(pair_idx['e'])}
        nuc_offsets = np.append(0, np.cumsum([len(pair_idx[t]) for t in nuc_keys]))
        if nuc_keys:
            npairs['n'] = int(nuc_offsets[-1])
        if all(on_gpu.values()):
            for p0, p1 in lib.prange(0, naux_slice, blksize):
                cderi_blk = {'e': cderi_sparse['e'][p0:p1]}
                if nuc_keys:
                    cderi_blk['n'] = cderi_sparse['n'][p0:p1]
                    for i, t in enumerate(nuc_keys):
                        q0, q1 = nuc_offsets[i:i+2]
                        cderi_blk[t] = cderi_blk['n'][:,q0:q1]
                if not unpack:
                    yield None, cderi_blk
                else:
                    out = {}
                    out_t = fill_symmetric(cderi_sparse['e'].T, pair_idx['e'],
                                           nao['e'], p0, p1, out=work['e'][:,:,:p1-p0])
                    out['e'] = out_t.transpose(2,0,1)
                    for i, t in enumerate(nuc_keys):
                        q0, q1 = nuc_offsets[i:i+2]
                        cderi_sparse_t = cderi_sparse['n'][:,q0:q1]
                        out_t = fill_symmetric(cderi_sparse_t.T, pair_idx[t],
                                               nao[t], p0, p1, out=work[t][:,:,:p1-p0])
                        out[t] = out_t.transpose(2,0,1)
                    yield out, cderi_blk

        else:
            buf = {}
            buf_prefetch = {}
            for t in storage_keys:
                if on_gpu[t]:
                    buf[t] = buf_prefetch[t] = None
                else:
                    buf[t] = cp.empty((blksize, npairs[t]))
                    buf_prefetch[t] = cp.empty_like(buf[t])

            comput_stream = cp.cuda.get_current_stream()
            compute_event = cp.cuda.Event()
            io_stream = cp.cuda.stream.Stream(non_blocking=True)
            io_event = cp.cuda.Event()

            for t in storage_keys:
                if on_gpu[t]:
                    buf_prefetch[t] = cderi_sparse[t][:blksize]
                else:
                    buf_prefetch[t].set(cderi_sparse[t][:blksize], stream=io_stream)
            io_event.record(io_stream)

            for p0, p1 in lib.prange(0, naux_slice, blksize):
                compute_event.record(comput_stream)
                buf, buf_prefetch = buf_prefetch, buf
                cderi_blk = {'e': buf['e'][:p1-p0]}
                if nuc_keys:
                    cderi_blk['n'] = buf['n'][:p1-p0]
                    for i, t in enumerate(nuc_keys):
                        q0, q1 = nuc_offsets[i:i+2]
                        cderi_blk[t] = cderi_blk['n'][:,q0:q1]
                comput_stream.wait_event(io_event)

                # prefetch the next block
                p2 = min(naux_slice, p1 + blksize)
                if p1 < p2:
                    io_stream.wait_event(compute_event)
                    for t in storage_keys:
                        if on_gpu[t]:
                            buf_prefetch[t] = cderi_sparse[t][p1:p2]
                        else:
                            buf_prefetch[t][:p2-p1].set(cderi_sparse[t][p1:p2], stream=io_stream)
                    io_event.record(io_stream)

                if not unpack:
                    yield None, cderi_blk
                else:
                    out = {}
                    for t in self.mol.components:
                        out_t = fill_symmetric(cderi_blk[t].T, pair_idx[t],
                                               nao[t], 0, p1-p0,
                                               out=work[t][:,:,:p1-p0])
                        out[t] = out_t.transpose(2,0,1)
                    yield out, cderi_blk

    def reset(self, mol=None):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        super().reset(mol)
        self.nao = {}
        self.intopt = {}
        self.j_engine = {}
        self._cderi_idx = {}
        self._cd_j2c = None
        self._auxmol_atom_major = None
        return self

    def to_cpu(self):
        return df_cpu.DF(self.mol, self.auxbasis,
                         df_ne_scheme=self.df_ne_scheme,
                         df_nn=self.df_nn,
                         nuc_auxbasis=self.nuc_auxbasis,
                         nuc_auxbasis_beta=self.nuc_auxbasis_beta,
                         nuc_auxbasis_lmax=self.nuc_auxbasis_lmax,
                         df_ne_component_vint=self.df_ne_component_vint)

def _decompose_j2c_schur(orig_auxmol, auxmol_e, sorted_auxmol, aux_sorting, omega=None):
    naux_e_ao = auxmol_e.nao
    j2c = int3c2e_bdiv.int2c2e(orig_auxmol, omega=omega)
    j2c_e = j2c[:naux_e_ao,:naux_e_ao]
    j2c_en = j2c[:naux_e_ao,naux_e_ao:]
    j2c_nn = j2c[naux_e_ao:,naux_e_ao:]
    low_e = cholesky(j2c_e)
    naux_e = low_e.shape[1]
    low_en = solve_triangular(low_e, j2c_en, lower=True, overwrite_b=True)
    j2c_nn -= low_en.T.dot(low_en)
    low_n = cholesky(j2c_nn)

    aux_coef = sorted_auxmol.ctr_coeff
    aux_coef[:,:naux_e_ao] = solve_triangular(
        low_e, aux_coef[:,:naux_e_ao].T, lower=True, overwrite_b=True).T
    aux_coef[:,naux_e_ao:] -= aux_coef[:,:naux_e_ao].dot(low_en)
    aux_coef[:,naux_e_ao:] = solve_triangular(
        low_n, aux_coef[:,naux_e_ao:].T, lower=True, overwrite_b=True).T

    if aux_sorting is not None:
        aux_coef, tmp = cp.empty_like(aux_coef), aux_coef
        aux_coef[aux_sorting] = tmp
    return aux_coef, 'cd', naux_e


def _cholesky_eri(intopt, auxmol_e=None, orig_auxmol=None,
                  omega=None, use_gpu_memory=None):
    cderi = {}
    cderi_idx = {}
    naux_e = None
    if 'n' in intopt or auxmol_e is not None:
        cderi['e'], cderi_idx['e'], aux_coef, naux_e = _cholesky_eri_with_aux_coef(
            intopt['e'], auxmol_e=auxmol_e, orig_auxmol=orig_auxmol,
            omega=omega, use_gpu_memory=use_gpu_memory)
    else:
        cderi['e'], cderi_idx['e'] = df._cholesky_eri(intopt['e'], omega=omega,
                                                      use_gpu_memory=use_gpu_memory)
        return cderi, cderi_idx, naux_e
    if 'n' in intopt:
        cderi['n'], cderi_idx_n, _, _ = _cholesky_eri_with_aux_coef(
            intopt['n'], aux_coef=aux_coef, omega=omega,
            use_gpu_memory=use_gpu_memory)
        cderi_idx.update(cderi_idx_n)
    return cderi, cderi_idx, naux_e

def _cholesky_eri_with_aux_coef(intopt, auxmol_e=None, aux_coef=None,
                                orig_auxmol=None, omega=None,
                                use_gpu_memory=None):
    assert isinstance(intopt, int3c2e_bdiv.Int3c2eOpt)
    if intopt._int3c2e_envs is None:
        intopt.build()
    sorted_mol = intopt.mol
    mol = sorted_mol.mol
    log = logger.new_logger(mol)
    auxmol = intopt.auxmol
    naux_sorted = auxmol.nao
    num_devices = multi_gpu.num_devices

    # When the basis set does not contain general contractions, sorted_mol are
    # simply an reordering of the original mol bases.
    recontract_bas = cp.asnumpy(sorted_mol.recontract_bas)
    needs_recontraction = any(recontract_bas[:,int3c2e_bdiv.NCTR_OF] != 1)

    mem_avail = get_avail_mem(exclude_memory_pool=True)
    word_avail = mem_avail // 8
    batch_size = int(word_avail * .2) // naux_sorted

    current_device = cp.cuda.device.get_device_id()
    eval_j3c, aux_sorting, ao_pair_offsets, _, clone_context = intopt.int3c2e_evaluator(
        ao_pair_batch_size=batch_size, reorder_aux=True,
        pair_batch_by_l=needs_recontraction, return_clone_context=True,
        omega=omega)
    cderi_batch_size = int(max(ao_pair_offsets[1:] - ao_pair_offsets[:-1]))
    batch_size = cderi_batch_size
    # * When the get_avail_mem() returns a small amount of memory, the actual
    #   size (cderi_batch_size) can be larger than the input batch_size.
    #   This cderi_batch_size should not be used to initialize eval_j3c for
    #   other devices. It may produce a different batches patterns.
    # * When multi-gpu is enabled, clone_context is used to clone eval_j3c on
    #   different GPUs, to ensure the same offsets, and same pair_addresses are
    #   created across devices.
    num_batches = len(ao_pair_offsets) - 1

    if needs_recontraction:
        if isinstance(intopt, neo_int3c2e_bdiv.Int3c2eOpt):
            recontract, ao_pair_counts, contracted_ao_pair_counts, cderi_idx = \
                    neo_int3c2e_bdiv._create_pair_recontractor(intopt, clone_context)
            cderi_npairs = sum(len(pair_addresses)
                               for pair_addresses, diag_addrs in cderi_idx.values())
        else:
            recontract, ao_pair_counts, contracted_ao_pair_counts, pair_addresses = \
                    int3c2e_bdiv._create_pair_recontractor(sorted_mol, clone_context)
            cderi_npairs = len(pair_addresses)
            pair_addresses = asarray(pair_addresses)
            rows, cols = divmod(pair_addresses, mol.nao)
            diag_addrs = cp.where(rows == cols)[0]
            cderi_idx = (pair_addresses, diag_addrs)
        cderi_offsets = np.append(0, np.cumsum(contracted_ao_pair_counts))
        cderi_batch_size = int(max(contracted_ao_pair_counts))
    else:
        cderi_offsets = ao_pair_offsets
        if isinstance(intopt, neo_int3c2e_bdiv.Int3c2eOpt):
            cderi_idx = intopt.pair_and_diag_indices()
            cderi_npairs = sum(len(pair_addresses)
                               for pair_addresses, diag_addrs in cderi_idx.values())
        else:
            pair_addresses, diag_addrs = intopt.pair_and_diag_indices()
            cderi_npairs = len(pair_addresses)
            pair_addresses = cp.asarray(pair_addresses, dtype=np.int32)
            cderi_idx = (pair_addresses, diag_addrs)

    naux_e = None
    if aux_coef is None:
        if auxmol_e is None:
            aux_coef, tag = df._decompose_j2c(auxmol, aux_sorting, omega)
        else:
            if orig_auxmol is None:
                raise RuntimeError('Original auxiliary molecule is required for Schur DF-NE')
            try:
                aux_coef, tag, naux_e = _decompose_j2c_schur(orig_auxmol, auxmol_e, auxmol,
                                                             aux_sorting, omega)
            except RuntimeError:
                aux_coef, tag = df._decompose_j2c(auxmol, aux_sorting, omega)
    if num_devices > 1:
        # cupy cannot copy non-contiguous array aux_coef[:,aux0:aux1] between
        # devices. This slicing is a contiguous array in the F-order storage.
        aux_coef = cp.asarray(aux_coef, order='F')

    naux = aux_coef.shape[1]
    naux_per_device = min(naux, (naux + num_devices - 1) // num_devices)

    cp.get_default_memory_pool().free_all_blocks()
    word_avail -= batch_size * naux_sorted
    if needs_recontraction:
        word_avail -= batch_size * naux_per_device

    # Put cderi on GPU whenever possible
    on_gpu = True
    if use_gpu_memory is False:
        on_gpu = False
    elif cderi_npairs * naux > word_avail * 0.95 * num_devices:
        if use_gpu_memory:
            cderi_size = cderi_npairs * naux / num_devices * 8e-9
            raise MemoryError(f'Not enough GPU memory. cderi size = {cderi_size:.2f} GB')
        on_gpu = False
    log.debug1('mem_avail=%.3f MB on_gpu=%s, nao_pairs=%d, naux_per_device=%d, batch_size=%d, num_batches=%d',
               mem_avail, on_gpu, cderi_npairs, naux_per_device, batch_size, num_batches)

    if not on_gpu:
        cderi_cpu = empty_mapped((naux, cderi_npairs))

        def proc(batch_iter):
            device_id = cp.cuda.device.get_device_id()
            c = cp.asarray(aux_coef)
            _eval_j3c = eval_j3c
            if device_id != current_device:
                _eval_j3c = intopt.int3c2e_evaluator(
                    reorder_aux=True, clone_context=clone_context, omega=omega)[0]

            work = cp.empty(naux_sorted * batch_size)
            work2 = cp.empty(naux * batch_size)
            if needs_recontraction:
                work1 = cp.empty(naux * cderi_batch_size)

            for batch_id in batch_iter:
                log.debug1('processing cderi batch %d', batch_id)
                j3c = _eval_j3c(shl_pair_batch_id=batch_id, out=work)
                cderi_gpu = ndarray((j3c.shape[0], naux), buffer=work2)
                cderi_gpu = j3c.dot(c, out=cderi_gpu)
                if needs_recontraction:
                    cderi_gpu = recontract(batch_id, cderi_gpu, out=work1)

                p0, p1 = cderi_offsets[batch_id:batch_id+2]
                # TODO: async-write to host memory in another stream
                err = libvhf_rys.transpose_write(
                    ctypes.cast(cp.cuda.get_current_stream().ptr, ctypes.c_void_p),
                    cderi_cpu.ctypes,
                    ctypes.cast(cderi_gpu.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(naux), ctypes.c_int(cderi_npairs),
                    ctypes.c_int(p0), ctypes.c_int(p1))
                if err != 0:
                    raise RuntimeError('transpose_write cderi_cpu failed')

        batch_iter = iter(range(num_batches))
        multi_gpu.run(proc, args=(batch_iter,), non_blocking=True)

        # Ensure data are fully written to host memory.
        multi_gpu.synchronize()
        cderi = [cderi_cpu[i*naux_per_device:(i+1)*naux_per_device]
                 for i in range(num_devices)]

    else:
        def proc():
            device_id = cp.cuda.device.get_device_id()
            aux0 = naux_per_device * device_id
            aux1 = min(naux, aux0 + naux_per_device)
            c = cp.asarray(aux_coef[:,aux0:aux1])

            _eval_j3c = eval_j3c
            if device_id != current_device:
                _eval_j3c = intopt.int3c2e_evaluator(
                    reorder_aux=True, clone_context=clone_context, omega=omega)[0]

            out = cp.empty((cderi_npairs, aux1-aux0))
            work = cp.empty(naux_sorted * batch_size)
            if needs_recontraction:
                work1 = cp.empty(naux_per_device * batch_size)

            for batch_id in range(num_batches):
                j3c = _eval_j3c(batch_id, out=work)
                p0, p1 = cderi_offsets[batch_id:batch_id+2]
                if needs_recontraction:
                    tmp = ndarray((j3c.shape[0], aux1-aux0), buffer=work1)
                    tmp = j3c.dot(c, out=tmp)
                    recontract(batch_id, tmp, out=out[p0:p1])
                else:
                    j3c.dot(c, out=out[p0:p1])
            return out.T

        cderi = multi_gpu.run(proc, non_blocking=True)
    return cderi, cderi_idx, aux_coef, naux_e


def get_jk(dfobj, dms, hermi=0, with_j=True, with_k=True,
           omega=None, lr_factor=None, sr_factor=None, output_components=None):
    if with_k:
        raise ValueError(
            'NEO DF global scheme builds J only; electronic K is handled '
            'by the electronic component')
    if not with_j:
        return None, None
    assert omega is None or abs(omega) < 1e-10
    assert lr_factor is None and sr_factor is None
    if dfobj.df_ne_component_vint:
        raise NotImplementedError('Caching component vint is not supported')
    if dfobj.df_ne_scheme == 'electron':
        raise NotImplementedError('df_ne_scheme="electron" is not implemented')
    if dfobj.df_ne_scheme != 'global':
        raise ValueError(f'Unsupported df_ne_scheme {dfobj.df_ne_scheme}')
    assert 'e' in dms
    if output_components is None:
        assert dms.keys() == dfobj.mol.components.keys()

    log = logger.new_logger(dfobj.mol, dfobj.verbose)
    t1 = t0 = log.init_timer()
    if dfobj._cderi is None:
        log.debug('Build CDERI ...')
        try:
            dfobj.build(omega=omega)
        except RuntimeError as err:
            logger.warn(dfobj, 'Global DF-NE CDERI build failed; switch to direct DF-J. %s', err)
            dfobj.df_ne_j_engine = 'direct'
            return get_j(dfobj, dms, hermi, output_components=output_components), None
        t1 = log.timer_debug1('init neo-j', *t0)

    out_cupy = isinstance(dms['e'], cupy.ndarray)
    if output_components is None:
        output_components = dfobj.mol.components
    else:
        assert all(t.startswith('n') for t in output_components)
    output_components = list(output_components)
    nuc_keys = [t for t in dfobj.mol.components if t.startswith('n')]
    nuc_outputs = [t for t in output_components if t.startswith('n')]
    nuc_source_keys = [t for t in nuc_keys if t in dms]
    assert not nuc_source_keys or len(nuc_source_keys) == len(nuc_keys)
    group_n_output = not dfobj.df_nn and nuc_outputs == nuc_keys

    dm_e = cupy.asarray(dms['e'])
    leading_shape = dm_e.shape[:-2]
    dms_3d = {}
    for t in dms:
        dm_t = cupy.asarray(dms[t])
        nao = dm_t.shape[-1]
        assert nao == dfobj.nao[t]
        dms_3d[t] = dm_t.reshape(-1, nao, nao)
    n_dm = dms_3d['e'].shape[0]
    for t in dms:
        if dms_3d[t].shape[0] != n_dm:
            raise ValueError('All component density matrices must have the same number of sets')
    out_shape = {t: leading_shape + (dfobj.nao[t], dfobj.nao[t]) for t in output_components}

    dm_sparse = {}
    rows = {}
    cols = {}
    for t in dfobj.mol.components:
        if t not in dms and t not in output_components:
            continue
        nao = dfobj.nao[t]
        pair_addresses, diags = dfobj._cderi_idx[t]
        rows[t], cols[t] = divmod(cp.asarray(pair_addresses), nao)
        if t not in dms:
            continue
        dm_sparse_t = dms_3d[t][:,rows[t],cols[t]]
        if hermi == 0:
            dm_sparse_t += dms_3d[t][:,cols[t],rows[t]]
        else:
            dm_sparse_t *= 2
        dm_sparse_t[:,diags] *= .5
        dm_sparse[t] = dm_sparse_t

    def proc():
        _dm_sparse = {t: cp.asarray(dm_sparse[t]) for t in dms}
        vj = {t: cp.zeros((n_dm, len(rows[t]))) for t in output_components}
        if group_n_output:
            vj_n = cp.zeros((n_dm, sum(len(rows[t]) for t in nuc_outputs)))

        blksize = dfobj.get_blksize(mem_fraction=0.4, nao=dfobj.nao, unpack=False)
        if nuc_source_keys:
            dm_n = cp.concatenate([_dm_sparse[t] * dfobj._charges[t]
                                   for t in nuc_source_keys], axis=1)

        for _, cderi_tril in dfobj.loop(blksize=blksize, unpack=False):
            cderi_e_blk = cderi_tril['e']
            rhoj_e_blk = contract('np,Lp->nL', _dm_sparse['e'], cderi_e_blk)
            rhoj_e_blk *= dfobj._charges['e']
            if nuc_source_keys:
                rhoj_n_blk = contract('np,Lp->nL', dm_n, cderi_tril['n'])
                rhoj_total_blk = rhoj_e_blk + rhoj_n_blk
            else:
                rhoj_total_blk = rhoj_e_blk
            for t in output_components:
                if t == 'e':
                    rhoj_t = rhoj_total_blk
                elif group_n_output:
                    continue
                elif dfobj.df_nn and t in dms:
                    rhoj_t = contract('np,Lp->nL', _dm_sparse[t], cderi_tril[t])
                    rhoj_t *= dfobj._charges[t]
                    rhoj_t = rhoj_total_blk - rhoj_t
                elif dfobj.df_nn:
                    rhoj_t = rhoj_total_blk
                else:
                    rhoj_t = rhoj_e_blk
                contract('nL,Lp->np', rhoj_t, cderi_tril[t], beta=1, out=vj[t])
            if group_n_output:
                contract('nL,Lp->np', rhoj_e_blk, cderi_tril['n'], beta=1, out=vj_n)

        if group_n_output:
            p0 = 0
            for t in nuc_outputs:
                p1 = p0 + len(rows[t])
                vj[t] = vj_n[:,p0:p1]
                p0 = p1
        for t in output_components:
            vj[t] *= dfobj._charges[t]
        return vj

    results = multi_gpu.run(proc, non_blocking=True)

    vj = {}
    for t in output_components:
        vj_t_sparse = multi_gpu.array_reduce([x[t] for x in results], inplace=True)
        vj_t = cp.zeros((n_dm, dfobj.nao[t], dfobj.nao[t]))
        vj_t[:,cols[t],rows[t]] = vj_t[:,rows[t],cols[t]] = vj_t_sparse
        vj[t] = vj_t.reshape(out_shape[t])
        if t.startswith('n'):
            vj[t] = tag_array(vj[t], vint=vj[t])
    if not out_cupy:
        vj = {t: v.get() for t, v in vj.items()}
    t1 = log.timer_debug1('neo-vj', *t1)
    return vj, None


def _get_rhoj(rhoj_e, rhoj_n, charges, components, df_nn):
    '''Assemble component source densities for the global NEO Coulomb term.'''
    rhoj_total = rhoj_e
    for t in rhoj_n:
        rhoj_total = rhoj_total + rhoj_n[t] * charges[t]
    rhoj = {'e': rhoj_total}
    for t in components:
        if t == 'e':
            continue
        if df_nn:
            if t in rhoj_n:
                rhoj[t] = rhoj_total - rhoj_n[t] * charges[t]
            else:
                rhoj[t] = rhoj_total
        else:
            rhoj[t] = rhoj_e
    return rhoj


def get_j(dfobj, dm, hermi=1, output_components=None):
    from gpu4pyscf.df.int3c2e_bdiv import int2c2e
    if dfobj.df_ne_component_vint:
        raise NotImplementedError('GPU DF-NE does not cache component vint')
    assert 'e' in dm
    if output_components is None:
        assert dm.keys() == dfobj.mol.components.keys()
    nuc_keys = [t for t in dfobj.mol.components if t.startswith('n')]
    if (dfobj.auxmol is None or not dfobj.nao or
        'e' not in dfobj.j_engine or
        (nuc_keys and 'n' not in dfobj.j_engine)):
        dfobj.build(build_cderi=False)

    if dfobj._cd_j2c is None:
        j2c = int2c2e(dfobj.auxmol)
        try:
            dfobj._cd_j2c = cholesky(j2c), 'cd'
        except RuntimeError:
            dfobj._cd_j2c = j2c, None

    if output_components is None:
        output_components = dfobj.mol.components
    else:
        assert all(t.startswith('n') for t in output_components)
    output_components = list(output_components)
    nuc_source_keys = [t for t in nuc_keys if t in dm]
    assert not nuc_source_keys or len(nuc_source_keys) == len(nuc_keys)

    dm_e = cupy.asarray(dm['e'])
    leading_shape = dm_e.shape[:-2]
    dm_shape = {}
    n_dm = None
    for t in dm:
        dm_t = cupy.asarray(dm[t])
        nao = dm_t.shape[-1]
        dm_shape[t] = dm_t.shape
        n_dm_t = dm_t.reshape(-1, nao, nao).shape[0]
        if n_dm is None:
            n_dm = n_dm_t
        elif n_dm_t != n_dm:
            raise ValueError('All component density matrices must have the same number of sets')
    for t in output_components:
        if t not in dm_shape:
            dm_shape[t] = leading_shape + (dfobj.nao[t], dfobj.nao[t])

    intopt_e = dfobj.j_engine['e']
    dm_e = intopt_e.mol.apply_C_mat_CT(dm_e)
    rhoj_e = intopt_e.contract_dm(dm_e, hermi)
    rhoj_e = intopt_e.auxmol.apply_CT_dot(rhoj_e, axis=-1)

    intopt_n = dfobj.j_engine['n']
    rhoj_n = {}
    if nuc_source_keys:
        dm_nuc = {t: intopt_n.component_opts[t].mol.apply_C_mat_CT(dm[t])
                  for t in nuc_source_keys}
        rhoj_n = intopt_n.contract_dm(dm_nuc, hermi)
        for t in nuc_source_keys:
            rhoj_n[t] = intopt_n.auxmol.apply_CT_dot(rhoj_n[t], axis=-1)
    rhoj_component = _get_rhoj(rhoj_e, rhoj_n, dfobj._charges, dfobj.mol.components, dfobj.df_nn)

    j2c, tag = dfobj._cd_j2c
    naux = rhoj_e.shape[-1]
    rhoj = []
    rhoj_keys = []
    if 'e' in output_components:
        rhoj.append(rhoj_component['e'].reshape(-1, naux))
        rhoj_keys.append('e')
    nuc_outputs = [t for t in output_components if t.startswith('n')]
    if nuc_outputs:
        if dfobj.df_nn:
            for t in nuc_outputs:
                rhoj.append(rhoj_component[t].reshape(-1, naux))
                rhoj_keys.append(t)
        else:
            rhoj.append(rhoj_component[nuc_outputs[0]].reshape(-1, naux))
            rhoj_keys.append('n')
    rhoj = rhoj[0] if len(rhoj) == 1 else cupy.vstack(rhoj)
    if tag == 'cd':
        rhoj = solve_triangular(j2c, rhoj.T, lower=True)
        rhoj = solve_triangular(j2c.T, rhoj, lower=False).T
    else:
        rhoj = cp.linalg.solve(j2c, rhoj.T).T
    n_rhs = n_dm
    rhoj = {t: rhoj[i*n_rhs:(i+1)*n_rhs] for i, t in enumerate(rhoj_keys)}

    vj = {}
    if 'e' in output_components:
        rhoj_t = intopt_e.auxmol.apply_C_dot(rhoj['e'], axis=-1)
        vj['e'] = intopt_e.contract_auxvec(rhoj_t)
        vj['e'] = intopt_e.mol.apply_CT_mat_C(vj['e'])
    if nuc_outputs:
        if dfobj.df_nn and nuc_source_keys:
            rhoj_n = {t: intopt_n.auxmol.apply_C_dot(rhoj[t], axis=-1) for t in nuc_keys}
            vj_n = intopt_n.contract_auxvec(rhoj_n, componentwise=True)
        else:
            rhoj_n = rhoj[nuc_outputs[0]] if dfobj.df_nn else rhoj['n']
            rhoj_n = intopt_n.auxmol.apply_C_dot(rhoj_n, axis=-1)
            vj_n = intopt_n.contract_auxvec(rhoj_n)
        for t in nuc_outputs:
            vj[t] = intopt_n.component_opts[t].mol.apply_CT_mat_C(vj_n[t])
    for t in output_components:
        vj[t] = cupy.asarray(vj[t]).reshape(dm_shape[t])
        vj[t] *= dfobj._charges[t]
        if t.startswith('n'):
            vj[t] = tag_array(vj[t], vint=vj[t])
    return vj


def density_fit(mf, auxbasis=None, with_df=None, ee_only_dfj=False,
                df_ne=True, df_nn=True, df_ne_scheme='global',
                nuc_auxbasis=None, nuc_auxbasis_beta=2.0,
                nuc_auxbasis_lmax=None,
                df_ne_component_vint=False, df_ne_j_engine='direct'):
    '''Apply density fitting to multicomponent Coulomb terms.

    If ``df_ne`` is false, only the electronic e-e Coulomb build is density
    fitted.  If ``df_ne`` is true, the DF tensor also covers
    electron-nuclear Coulomb interactions.

    ``df_nn`` fits distinguishable quantum-nuclear n-n Coulomb terms with the
    same global DF metric and defaults to true.  Setting ``df_ne=False``
    disables both electron-nuclear and nuclear-nuclear density fitting.

    ``df_ne_scheme='global'`` is the implemented DF-NE scheme.  It builds
    one mixed auxiliary metric for the electronic and nuclear auxiliary
    functions and uses the same transformed tensor for e-e and e-n Coulomb
    builds.  The default nuclear auxiliary basis is generated by the
    ``aug_etb`` recipe with the exponent-sum range, which targets AO-product
    densities rather than AO functions.  Its maximum angular momentum matches
    the nuclear orbital basis rather than using the element's electronic
    configuration.

    ``df_ne_scheme='electron'`` is accepted by the wrapper for API symmetry,
    but it is not implemented here.

    ``nuc_auxbasis`` controls only the nuclear auxiliary functions in the
    global scheme.  Named nuclear bases such as ``'pb4d'`` can be used, but
    they are generally not recommended as fitting bases because they were
    designed for nuclear orbitals instead of nuclear density products.
    Explicit even-tempered strings such as ``'8s8p8d'`` are also accepted; for
    these manual nuclear auxiliary bases the starting exponent is doubled
    relative to the NEO AO basis generator to match the equal-exponent product
    scale.  ``nuc_auxbasis_beta`` controls the spacing of the default
    ``aug_etb`` nuclear auxiliary basis.  ``nuc_auxbasis_lmax`` controls its
    maximum angular momentum and defaults to that of the nuclear AO basis.

    ``df_ne_j_engine='direct'`` uses the two-pass on-the-fly 3c DF-J builder
    and is the default path.  ``df_ne_j_engine='cderi'`` uses stored CDERI
    tensors and can build the electronic DF-K tensor as a Schur byproduct.

    ``df_ne_component_vint`` controls an extra compatibility path for
    component-level calls such as ``mf.components['e'].get_fock()`` after a
    DF-NE calculation.  This implementation currently does not cache this
    component vint and raises at the J build or SCF effective-potential build
    if it is requested.
    '''
    assert isinstance(mf, hf.HF)
    assert 'e' in mf.components
    assert isinstance(mf.components['e'], scf.hf.SCF)
    if df_ne and df_ne_scheme not in ('electron', 'global'):
        raise ValueError(f'Unsupported df_ne_scheme {df_ne_scheme}')
    if not df_ne:
        df_nn = False
    if df_ne_j_engine not in ('cderi', 'direct'):
        raise ValueError(f'Unsupported df_ne_j_engine {df_ne_j_engine}')

    if with_df is None and df_ne:
        # e-e and e-n with_df
        with_df = DF(mf.mol, auxbasis, df_ne_scheme=df_ne_scheme,
                     nuc_auxbasis=nuc_auxbasis,
                     nuc_auxbasis_beta=nuc_auxbasis_beta,
                     nuc_auxbasis_lmax=nuc_auxbasis_lmax,
                     df_nn=df_nn,
                     df_ne_component_vint=df_ne_component_vint,
                     df_ne_j_engine=df_ne_j_engine)

    if with_df is not None and df_ne:
        if not isinstance(with_df, DF):
            raise TypeError('with_df must be neo.df.DF when df_ne=True')
        if with_df.mol is not mf.mol:
            if with_df._cderi:
                raise ValueError('A built with_df object cannot be reused for a different NEO mol')
            with_df.reset(mf.mol)
        with_df.df_ne_scheme = df_ne_scheme
        with_df.nuc_auxbasis = nuc_auxbasis
        with_df.nuc_auxbasis_beta = nuc_auxbasis_beta
        with_df.nuc_auxbasis_lmax = nuc_auxbasis_lmax
        with_df.df_ne_component_vint = df_ne_component_vint
        with_df.df_ne_j_engine = df_ne_j_engine
        with_df.df_nn = df_nn
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df._charges.clear()
        for t, comp in mf.components.items():
            with_df._charges[t] = comp.charge
            if isinstance(comp, scf.rohf.ROHF):
                raise NotImplementedError
    elif with_df is not None:
        assert isinstance(with_df, df.DF) and not isinstance(with_df, DF)
        if isinstance(mf.components['e'], df_jk._DFHF):
            if mf.components['e'].with_df is None:
                mf.components['e'].with_df = with_df
            elif getattr(mf.components['e'].with_df, 'auxbasis', None) != auxbasis:
                mf = mf.copy()
                mf.components['e'].with_df = with_df
                mf.components['e'].only_dfj = ee_only_dfj
            return mf

    if isinstance(mf, _DFNEO):
        if mf.with_df is None:
            mf.with_df = with_df
        elif getattr(mf.with_df, 'auxbasis', None) != auxbasis:
            mf = mf.copy()
            mf.with_df = with_df
            mf.ee_only_dfj = ee_only_dfj
        mf.df_ne = df_ne
        mf.df_nn = df_nn
        mf.df_ne_component_vint = df_ne_component_vint

    _charge = mf.components['e'].charge
    _mass = mf.components['e'].mass
    _is_nucleus = mf.components['e'].is_nucleus
    _nuc_occ_state = mf.components['e'].nuc_occ_state
    base = mf.components['e'].undo_component()
    if isinstance(base, df_jk._DFHF):
        base = base.undo_df()
    if with_df is not None:
        auxbasis = with_df.auxbasis
    mf.components['e'] = hf.general_scf(
        df_jk.density_fit(base, auxbasis=auxbasis,
                          with_df=None if df_ne else with_df,
                          only_dfj=ee_only_dfj),
        charge=_charge, mass=_mass, is_nucleus=_is_nucleus,
        nuc_occ_state=_nuc_occ_state)

    if isinstance(with_df, DF):
        if with_df.df_ne_scheme == 'global':
            with_df._elec_with_df = mf.components['e'].with_df

    if isinstance(mf, ks.KS):
        mf.interactions = hf.hf_cpu.generate_interactions(
            mf.components, ks.InteractionCorrelation, mf.max_memory,
            mf.direct_scf_tol, epc=mf.epc)
    else:
        mf.interactions = hf.hf_cpu.generate_interactions(
            mf.components, hf.InteractionCoulomb, mf.max_memory,
            mf.direct_scf_tol)

    if isinstance(mf, _DFNEO):
        return mf

    dfmf = _DFNEO(mf, with_df, ee_only_dfj, df_ne, df_nn, df_ne_component_vint)
    return lib.set_class(dfmf, (_DFNEO, mf.__class__))


def from_cpu(mf):
    base = mf.undo_df().to_gpu()
    with_df = getattr(mf, 'with_df', None)
    if with_df is None:
        with_df = getattr(mf.components['e'], 'with_df', None)
    auxbasis = getattr(with_df, 'auxbasis', None)
    kwargs = {'auxbasis': auxbasis, 'ee_only_dfj': mf.ee_only_dfj,
              'df_ne': getattr(mf, 'df_ne', False)}
    if hasattr(mf, 'df_ne_component_vint'):
        kwargs['df_ne_component_vint'] = mf.df_ne_component_vint
    if hasattr(mf, 'df_nn'):
        kwargs['df_nn'] = mf.df_nn
    for key in ('df_ne_scheme', 'nuc_auxbasis', 'nuc_auxbasis_beta',
                'nuc_auxbasis_lmax', 'df_ne_j_engine'):
        if hasattr(with_df, key):
            kwargs[key] = getattr(with_df, key)
    return density_fit(base, **kwargs)


class _DFNEO:
    __name_mixin__ = 'DF'
    _keys = {'with_df', 'ee_only_dfj', 'df_ne', 'df_nn',
             'df_ne_component_vint'}

    def __init__(self, mf, with_df, ee_only_dfj, df_ne, df_nn, df_ne_component_vint):
        self.__dict__.update(mf.__dict__)
        self._eri = None
        self.with_df = with_df
        self.ee_only_dfj = ee_only_dfj
        self.df_ne = df_ne
        self.df_ne_component_vint = df_ne_component_vint
        self.df_nn = df_nn
        self.direct_scf = False

    def undo_df(self):
        obj = lib.view(self, lib.drop_class(self.__class__, _DFNEO))
        obj.components = {}
        for t, comp in self.components.items():
            if t == 'e':
                base = comp.undo_component().undo_df()
            else:
                base = comp.undo_component()
            obj.components[t] = hf.general_scf(base.copy(),
                                               charge=comp.charge,
                                               mass=comp.mass,
                                               is_nucleus=comp.is_nucleus,
                                               nuc_occ_state=comp.nuc_occ_state)
        if isinstance(obj, ks.KS):
            obj.interactions = hf.hf_cpu.generate_interactions(
                obj.components, ks.InteractionCorrelation, obj.max_memory,
                obj.direct_scf_tol, epc=obj.epc)
        else:
            obj.interactions = hf.hf_cpu.generate_interactions(
                obj.components, hf.InteractionCoulomb, obj.max_memory,
                obj.direct_scf_tol)
        if hasattr(self, 'f') and self.f is not None:
            obj.f = cp.array(self.f, copy=True)
        del obj.with_df, obj.ee_only_dfj, obj.df_ne, obj.df_nn, obj.df_ne_component_vint
        return obj

    def reset(self, mol=None):
        e_with_df = self.components['e'].with_df
        if self.with_df is not None:
            self.with_df.reset(mol)
        super().reset(mol)
        mf_e = self.components['e']
        if not isinstance(mf_e, df_jk._DFHF):
            # A spin change replaced the electronic component. Restore only its
            # ordinary DF wrapper, retaining the DF settings.
            e_with_df.reset(mf_e.mol)
            mf_e = hf.general_scf(df_jk.density_fit(mf_e.undo_component(),
                                                    with_df=e_with_df,
                                                    only_dfj=self.ee_only_dfj),
                                  charge=mf_e.charge, mass=mf_e.mass,
                                  is_nucleus=mf_e.is_nucleus,
                                  nuc_occ_state=mf_e.nuc_occ_state)
            self.components['e'] = mf_e
            # The parent already constructed the pair molecules and spin flags.
            # Only the electronic SCF reference changes when DF is restored.
            for interaction in self.interactions.values():
                if interaction.mf1_type == 'e':
                    interaction.mf1 = mf_e
                if interaction.mf2_type == 'e':
                    interaction.mf2 = mf_e
        if self.df_ne:
            self.with_df._charges.clear()
            for t, comp in self.components.items():
                self.with_df._charges[t] = comp.charge
            if self.with_df.df_ne_scheme == 'global':
                self.with_df._elec_with_df = mf_e.with_df
        return self

    def get_j(self, mol=None, dm=None, hermi=1, omega=None):
        return self.get_jk(mol, dm, hermi, with_j=True, with_k=False, omega=omega)[0]

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None, lr_factor=None, sr_factor=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        return self.with_df.get_jk(dm, hermi, with_j, with_k, omega=omega,
                                   lr_factor=lr_factor, sr_factor=sr_factor)

    def _get_init_guess_vint(self, output_components, dm_guess):
        if not self.with_df or not self.df_ne:
            return super()._get_init_guess_vint(output_components, dm_guess)

        assert 'e' in dm_guess
        dm_e = cupy.asarray(dm_guess['e'])
        if isinstance(self.components['e'], scf.uhf.UHF):
            dm_e = dm_e[0] + dm_e[1]
        mf_e = self.components['e']
        build_e_cderi = (self.with_df.df_ne_scheme == 'global' and
                         not self.ee_only_dfj and
                         (not isinstance(mf_e, scf.hf.KohnShamDFT) or
                          mf_e._numint.libxc.is_hybrid_xc(mf_e.xc)))
        with lib.temporary_env(self.with_df, _build_e_cderi=build_e_cderi):
            if self.with_df.df_ne_j_engine == 'direct':
                return get_j(self.with_df, {'e': dm_e}, output_components=output_components)
            return get_jk(self.with_df, {'e': dm_e}, with_k=False,
                          output_components=output_components)[0]

    def get_veff(self, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
        if not self.with_df or not self.df_ne:
            return super().get_veff(mol, dm, dm_last, vhf_last, hermi)
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()

        if self.with_df.df_ne_scheme == 'electron':
            raise NotImplementedError('GPU NEO df_ne_scheme="electron" is not implemented')
        if self.with_df.df_ne_scheme != 'global':
            raise ValueError(f'Unsupported df_ne_scheme {self.with_df.df_ne_scheme}')
        if self.df_ne_component_vint or self.with_df.df_ne_component_vint:
            raise NotImplementedError('GPU DF-NE does not cache component vint')

        assert not self.direct_scf
        mf_e = self.components['e']
        mol_e = mol.components['e']
        if isinstance(mf_e, (scf.rohf.ROHF, scf.ghf.GHF)):
            raise NotImplementedError

        log = logger.new_logger(self)

        vhf = {}
        epc = {t: 0 for t in self.components}
        # Evaluate electronic XC, NLC, and EPC before the global DF-J build.
        if isinstance(mf_e, scf.hf.KohnShamDFT):
            t0 = log.init_timer()
            ni = mf_e._numint
            dm_e = dm['e']
            if isinstance(mf_e, scf.uhf.UHF):
                if mf_e.grids.coords is None:
                    mf_e.initialize_grids(mol_e, dm_e[0]+dm_e[1])
                n, exc, vxc = ni.nr_uks(mol_e, mf_e.grids, mf_e.xc, dm_e)
                log.debug('nelec by numeric integration = %s', n)
                if mf_e.do_nlc():
                    if ni.libxc.is_nlc(mf_e.xc):
                        xc = mf_e.xc
                    else:
                        assert ni.libxc.is_nlc(mf_e.nlc)
                        xc = mf_e.nlc
                    n, enlc, vnlc = ni.nr_nlc_vxc(mol_e, mf_e.nlcgrids, xc, dm_e)
                    exc += enlc
                    vxc += vnlc
                    log.debug('nelec with nlc grids = %s', n)
                t0 = log.timer('e vxc', *t0)

                if getattr(self, 'epc', None) is not None:
                    epc = ks._get_epc_vmat(self, dm)
                    vxc += epc['e']
                    if hasattr(epc['e'], 'exc'):
                        exc += epc['e'].exc
                    t0 = log.timer('epc', *t0)

                build_e_cderi = (self.with_df.df_ne_j_engine == 'cderi' and
                                 self.with_df.df_ne_scheme == 'global' and
                                 not self.ee_only_dfj and
                                 mf_e._numint.libxc.is_hybrid_xc(mf_e.xc))
                _dm_tot = dm.copy()
                _dm_tot['e'] = dm_e[0] + dm_e[1]
                with lib.temporary_env(self.with_df, _build_e_cderi=build_e_cderi):
                    vj = self.get_j(mol, _dm_tot, hermi)
                t0 = log.timer('global vj', *t0)

                if not self.with_df.df_nn:
                    nn_vint = self._get_nn_vint(dm)
                    t0 = log.timer('n-n vj', *t0)
                    for t in vj:
                        vj[t] = vj[t] + nn_vint[t]

                if not ni.libxc.is_hybrid_xc(mf_e.xc):
                    vxc += vj['e']
                else:
                    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf_e.xc, spin=mol_e.spin)
                    # Global J is already available; only electronic K follows
                    # the original DF range-separated mixing mode.
                    if omega != 0:
                        range_separated_mode = getattr(mf_e, 'range_separated_mode', 'mix_outside_kernel')
                        if range_separated_mode == 'mix_outside_kernel':
                            vk = mf_e.get_k(mol_e, dm_e, hermi)
                            vk *= hyb
                            vklr = mf_e.get_k(mol_e, dm_e, hermi, omega=omega)
                            vklr *= (alpha - hyb)
                            vk += vklr
                        elif range_separated_mode == 'mix_inside_kernel':
                            vk = mf_e.get_k(mol_e, dm_e, hermi, omega=omega, lr_factor=alpha, sr_factor=hyb)
                        else:
                            raise ValueError(f'range_separated_mode = {range_separated_mode} is not supported')
                    else: # omega == 0
                        vk = mf_e.get_k(mol_e, dm_e, hermi)
                        vk *= hyb
                    vxc += vj['e'] - vk
                    exc -= float(cupy.einsum('sij,sji->', dm_e, vk).real.get()) * .5
                ecoul = float(cupy.einsum('sij,ji->', dm_e, vj['e']).real.get()) * .5

            elif isinstance(mf_e, scf.hf.RHF):
                mf_e.initialize_grids(mol_e, dm_e)
                n, exc, vxc = ni.nr_rks(mol_e, mf_e.grids, mf_e.xc, dm_e)
                log.debug('nelec by numeric integration = %s', n)
                if mf_e.do_nlc():
                    if ni.libxc.is_nlc(mf_e.xc):
                        xc = mf_e.xc
                    else:
                        assert ni.libxc.is_nlc(mf_e.nlc)
                        xc = mf_e.nlc
                    n, enlc, vnlc = ni.nr_nlc_vxc(mol_e, mf_e.nlcgrids, xc, dm_e)
                    exc += enlc
                    vxc += vnlc
                    log.debug('nelec with nlc grids = %s', n)
                t0 = log.timer('e vxc', *t0)

                if getattr(self, 'epc', None) is not None:
                    epc = ks._get_epc_vmat(self, dm)
                    vxc += epc['e']
                    if hasattr(epc['e'], 'exc'):
                        exc += epc['e'].exc
                    t0 = log.timer('epc', *t0)

                build_e_cderi = (self.with_df.df_ne_j_engine == 'cderi' and
                                 self.with_df.df_ne_scheme == 'global' and
                                 not self.ee_only_dfj and
                                 mf_e._numint.libxc.is_hybrid_xc(mf_e.xc))
                with lib.temporary_env(self.with_df, _build_e_cderi=build_e_cderi):
                    vj = self.get_j(mol, dm, hermi)
                t0 = log.timer('global vj', *t0)

                if not self.with_df.df_nn:
                    nn_vint = self._get_nn_vint(dm)
                    t0 = log.timer('n-n vj', *t0)
                    for t in vj:
                        vj[t] = vj[t] + nn_vint[t]

                if not ni.libxc.is_hybrid_xc(mf_e.xc):
                    vxc += vj['e']
                else:
                    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf_e.xc, spin=mol_e.spin)
                    # Global J is already available; only electronic K follows
                    # the original DF range-separated mixing mode.
                    if omega != 0:
                        range_separated_mode = getattr(mf_e, 'range_separated_mode', 'mix_outside_kernel')
                        if range_separated_mode == 'mix_outside_kernel':
                            vk = mf_e.get_k(mol_e, dm_e, hermi)
                            vk *= hyb
                            vklr = mf_e.get_k(mol_e, dm_e, hermi, omega=abs(omega))
                            vklr *= (alpha - hyb)
                            vk += vklr
                        elif range_separated_mode == 'mix_inside_kernel':
                            vk = mf_e.get_k(mol_e, dm_e, hermi, omega=omega, lr_factor=alpha, sr_factor=hyb)
                        else:
                            raise ValueError(f'range_separated_mode = {range_separated_mode} is not supported')
                    else: # omega == 0
                        vk = mf_e.get_k(mol_e, dm_e, hermi)
                        vk *= hyb
                    vxc += vj['e'] - vk * .5
                    exc -= float(cupy.einsum('ij,ji->', dm_e, vk).real.get()) * .25
                ecoul = float(cupy.einsum('ij,ji->', dm_e, vj['e']).real.get()) * .5
            else:
                raise NotImplementedError("DF only supports R/U KS.")
            t0 = log.timer('veff (excluding vj)', *t0)
            vhf['e'] = tag_array(vxc, ecoul=ecoul, exc=exc)
        else:
            build_e_cderi = (self.with_df.df_ne_j_engine == 'cderi' and
                             self.with_df.df_ne_scheme == 'global' and
                             not self.ee_only_dfj)
            if isinstance(mf_e, scf.uhf.UHF):
                _dm_tot = dm.copy()
                _dm_tot['e'] = dm['e'][0] + dm['e'][1]
            else:
                _dm_tot = dm
            with lib.temporary_env(self.with_df, _build_e_cderi=build_e_cderi):
                vj = self.get_j(mol, _dm_tot, hermi)

            if not self.with_df.df_nn:
                nn_vint = self._get_nn_vint(dm)
                for t in vj:
                    vj[t] = vj[t] + nn_vint[t]

            vk = mf_e.get_k(mol_e, dm['e'], hermi)
            if isinstance(mf_e, scf.uhf.UHF):
                vhf['e'] = vj['e'] - vk
                ecoul = float(cupy.einsum('sij,ji->', dm['e'], vj['e']).real.get()) * .5
            elif isinstance(mf_e, scf.hf.RHF):
                vhf['e'] = vj['e'] - vk * .5
                ecoul = float(cupy.einsum('ij,ji->', dm['e'], vj['e']).real.get()) * .5
            else:
                raise NotImplementedError("DF only supports R/U/RO/G HF.")
            vhf['e'] = tag_array(vhf['e'], ecoul=ecoul)

        self.components['e']._vint = None

        # As in hf._grouped_energy, batch equal-sized nuclear AO matrices
        # to contract and download their individual Coulomb energies together.
        groups = {}
        for t in self.components:
            if t != 'e' and dm[t].ndim == 2:
                key = (dm[t].shape, dm[t].dtype, vj[t].dtype)
                groups.setdefault(key, []).append(t)
        ecoul_n = {}
        for keys in groups.values():
            if len(keys) == 1:
                t = keys[0]
                ecoul_n[t] = float(cupy.einsum('ij,ji->', cupy.asarray(dm[t]), vj[t]).real.get()) * .5
            else:
                dm_batch = cupy.stack([cupy.asarray(dm[t]) for t in keys])
                vj_batch = cupy.stack([vj[t] for t in keys])
                energies = cupy.einsum('nij,nji->n', dm_batch, vj_batch).real.get() * .5
                ecoul_n.update(zip(keys, map(float, energies)))

        for t, comp in self.components.items():
            if t == 'e':
                continue

            assert isinstance(comp, scf.hf.RHF)
            assert not isinstance(comp, (scf.rohf.ROHF, scf.uhf.UHF, scf.ghf.GHF))
            vint_t = vj[t] + epc[t]
            comp._vint = cupy.asarray(vint_t)
            ecoul = ecoul_n.get(t)
            vhf[t] = tag_array(vint_t, ecoul=ecoul, vint=vint_t)
            if hasattr(epc[t], 'exc'):
                vhf[t] = tag_array(vhf[t], exc=epc[t].exc)
        return vhf

    def _get_nn_vint(self, dm):
        dm = hf._to_cpu({t: dm[t] for t in self.components if t.startswith('n')})
        out = {t: 0 for t in self.components}
        for t_pair, interaction in self.interactions.items():
            if 'e' in t_pair:
                continue
            v = interaction.get_vint(dm)
            for t, val in v.items():
                out[t] = out[t] + val
        for t, val in out.items():
            if isinstance(val, np.ndarray):
                out[t] = cupy.asarray(val)
        return out

    def nuc_grad_method(self):
        if not self.df_ne:
            from gpu4pyscf.neo import grad
            return grad.Gradients(self)
        from gpu4pyscf.neo import df_grad
        return df_grad.Gradients(self)

    Gradients = lib.alias(nuc_grad_method, alias_name='Gradients')

    def to_cpu(self):
        obj = self.undo_df().to_cpu()
        with_df = self.with_df or self.components['e'].with_df
        kwargs = {'auxbasis': with_df.auxbasis, 'ee_only_dfj': self.ee_only_dfj,
                  'df_ne': self.df_ne,
                  'df_nn': self.df_nn,
                  'df_ne_component_vint': self.df_ne_component_vint}
        for key in ('df_ne_scheme', 'nuc_auxbasis', 'nuc_auxbasis_beta', 'nuc_auxbasis_lmax'):
            if hasattr(with_df, key):
                kwargs[key] = getattr(with_df, key)
        return df_cpu.density_fit(obj, **kwargs)
