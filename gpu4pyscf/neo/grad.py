import copy
import ctypes
import math
import cupy
import numpy
import warnings
from scipy.special import erf
from pyscf import gto, lib, scf
from pyscf.gto import ATOM_OF
from pyscf.grad import rhf as rhf_grad_cpu
from gpu4pyscf.dft import numint
from gpu4pyscf.dft import rks
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.gto.ecp import get_ecp_ip
from gpu4pyscf.gto.mole import SortedGTO, SortedMole
from gpu4pyscf.gto.int3c1e_ip import int1e_grids_ip1, int1e_grids_ip2
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import add_sparse, condense
from gpu4pyscf.neo import ks
from gpu4pyscf.qmmm.itrf import _mm_charge_integrals


def general_grad(grad_method):
    if isinstance(grad_method, ComponentGrad):
        return grad_method
    return lib.set_class(ComponentGrad(grad_method), (ComponentGrad, grad_method.__class__))


class ComponentGrad:
    __name_mixin__ = 'Component'

    def __init__(self, grad_method):
        self.__dict__.update(grad_method.__dict__)

    def get_hcore(self, mol=None, exclude_ecp=False):
        from gpu4pyscf.pbc.gto.int1e import int1e_ipkin
        if mol is None: mol = self.mol
        if mol._pseudo:
            raise NotImplementedError('Nuclear gradients for GTH PP')

        if getattr(self.base, 'with_x2c', None):
            raise NotImplementedError('X2C gradients')

        sorted_mol = SortedMole.from_mol(mol, decontract=True)
        h = -int1e_ipkin(sorted_mol) / self.base.mass
        h -= rhf_grad.int1e_ipnuc(mol) * self.base.charge
        if not exclude_ecp and not self.base.is_nucleus and mol.has_ecp():
            h -= get_ecp_ip(mol).sum(axis=0) * self.base.charge
        mm_mol = getattr(getattr(mol, 'super_mol', None), 'mm_mol', None)
        if mm_mol is not None:
            h += _mm_charge_integrals(mm_mol, mol, int1e_grids_ip1) * self.base.charge
        return h

    def _hcore_energy(self, dm0, dme0):
        if not self.base.is_nucleus:
            return super()._hcore_energy(dm0, dme0)

        mol = self.mol
        if mol._pseudo:
            raise NotImplementedError("Pseudopotential gradient not supported for molecular system yet")

        dh = _grouped_hcore_energy({'n': self}, {'n': dm0}, {'n': dme0})

        return dh

    def energy_ee(self, mol=None, dm=None, verbose=None):
        if self.base.is_nucleus:
            if mol is None: mol = self.mol
            return numpy.zeros((mol.natm, 3))
        return super().energy_ee(mol, dm)

    def get_veff(self, mol=None, dm=None):
        if self.base.is_nucleus:
            if mol is None: mol = self.mol
            return cupy.zeros((3, mol.nao, mol.nao))
        assert abs(self.base.charge) == 1
        return super().get_veff(mol, dm)

    def extra_force(self, atom_id=None):
        assert atom_id is None
        mm_mol = getattr(getattr(self.mol, 'super_mol', None), 'mm_mol', None)
        if mm_mol is None:
            return super().extra_force()

        h1 = _mm_charge_integrals(mm_mol, self.mol, int1e_grids_ip1) * self.base.charge
        dm = self.base.make_rdm1()
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        e1_grad = rhf_grad.contract_h1e_dm(self.mol, h1, dm, hermi=1)
        e1_grad += super().extra_force()
        return e1_grad

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        raise AttributeError


rhf_grad.libvhf_rys.RYS_per_atom_j_ip1_multi_in.restype = ctypes.c_int
rhf_grad.libvhf_rys.RYS_per_atom_j_ip1_multi_in_no_self.restype = ctypes.c_int


def _j_intercomponent_energy_per_atom(vhfopt, mols, dms, group1_size, verbose=None):
    '''
    Computes the first-order derivatives of the intercomponent Coulomb energy
    per atom.
    '''
    # Evaluate both directions of the intercomponent Coulomb gradient.
    # Component masks select local density pairs and cross-component integral
    # pairs in the combined molecule.
    log = logger.new_logger(vhfopt.mol, verbose)
    cput0 = log.init_timer()
    mol = vhfopt.sorted_mol

    component_mols = [SortedGTO.from_mol(mol_t, decontract=True,
                                         diffuse_cutoff=0.3)
                      for mol_t in mols]
    dms = [mol_t.apply_C_mat_CT(cupy.asarray(dm_t, order='C'))
           for mol_t, dm_t in zip(component_mols, dms)]

    orig_shell_component = numpy.hstack([
        numpy.full(mol_t.nbas, i, dtype=numpy.int32)
        for i, mol_t in enumerate(component_mols)])
    orig_shell_local = numpy.hstack([
        numpy.arange(mol_t.nbas, dtype=numpy.int32)
        for mol_t in component_mols])
    shell_component = orig_shell_component[mol.sorted_idx]
    shell_local = orig_shell_local[mol.sorted_idx]
    local_ao_loc = numpy.empty(mol.nbas, dtype=numpy.int32)
    component_nao = numpy.asarray([mol_t.nao for mol_t in component_mols], dtype=numpy.int32)
    local_shell = []
    for i, mol_t in enumerate(component_mols):
        inv_sorted = numpy.empty_like(mol_t.sorted_idx)
        inv_sorted[mol_t.sorted_idx] = numpy.arange(mol_t.nbas)
        idx = numpy.where(shell_component == i)[0]
        shell_t = inv_sorted[shell_local[idx]]
        local_shell.append(shell_t)
        local_ao_loc[idx] = mol_t.ao_loc[shell_t]
    if group1_size is not None:
        shell_group = shell_component >= group1_size

    uniq_l_ctr = mol.uniq_l_ctr
    uniq_l = uniq_l_ctr[:,0]
    l_ctr_bas_loc = numpy.append(0, numpy.cumsum(mol.l_ctr_counts))
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
    assert uniq_l.max() <= rhf_grad.LMAX

    log_cutoff = math.log(vhfopt.direct_scf_tol)
    dm_penalty = 0
    diffuse_exps, diffuse_ctr_coef = rhf_grad.extract_pgto_params(mol, 'diffuse')
    n_groups = len(uniq_l_ctr)
    tasks = [(i, j, k, l)
             for i in range(n_groups)
             for j in range(i+1)
             for k in range(i+1)
             for l in range(k+1)]
    schemes = {t: rhf_grad._ejk_quartets_scheme(mol, uniq_l_ctr[list(t)]) for t in tasks}
    tasks = iter(tasks)

    def proc():
        device_id = cupy.cuda.device.get_device_id()
        log = logger.new_logger(mol, verbose)
        cput0 = log.init_timer()

        timing_collection = rhf_grad._TimingCollector(log.timer_debug1)
        kern_counts = 0
        if group1_size is None:
            kern = rhf_grad.libvhf_rys.RYS_per_atom_j_ip1_multi_in_no_self
        else:
            kern = rhf_grad.libvhf_rys.RYS_per_atom_j_ip1_multi_in

        _dms = [cupy.asarray(dm_t, order='C') for dm_t in dms]
        dm_ptrs = []
        for dm_t in _dms:
            dm_ptrs.append(dm_t.data.ptr)
        dm_ptrs = cupy.asarray(numpy.asarray(dm_ptrs, dtype=numpy.uintp))
        ejk = cupy.zeros((mol.natm, 3))
        dm_cond = cupy.full((mol.nbas, mol.nbas), -1e30, dtype=numpy.float32)
        for i, (dm_t, mol_t) in enumerate(zip(_dms, component_mols)):
            dm_cond_t = cupy.log(condense('absmax', dm_t[None], mol_t.ao_loc)
                                 + 1e-300).astype(numpy.float32)
            idx = numpy.where(shell_component == i)[0]
            dm_cond[numpy.ix_(idx, idx)] = dm_cond_t[numpy.ix_(local_shell[i], local_shell[i])]
        _diffuse_exps = cupy.asarray(diffuse_exps, dtype=numpy.float32)
        _shell_component = cupy.asarray(shell_component, dtype=numpy.int32)
        if group1_size is not None:
            _shell_group = cupy.asarray(shell_group)
        _local_ao_loc = cupy.asarray(local_ao_loc, dtype=numpy.int32)
        _component_nao = cupy.asarray(component_nao, dtype=numpy.int32)
        # VHFOpt has already removed cross-component density pairs before
        # overlap and Schwarz screening.
        bas_pair_cache = {k: [cupy.asarray(x) for x in v]
                          for k, v in vhfopt.bas_pair_cache.items()}
        rys_envs = vhfopt.rys_envs
        workers = rhf_grad.gpu_specs['multiProcessorCount']
        # An additional integer to count for the proccessed pair_ijs
        pool = cupy.empty(workers*rhf_grad.QUEUE_DEPTH+1, dtype=numpy.int32)
        dd_pool = cupy.empty((workers, rhf_grad.DD_CACHE_MAX), dtype=numpy.float64)
        t1 = log.timer_debug1(f'q_cond and dm_cond on Device {device_id}', *cput0)

        for task in tasks:
            i, j, k, l = task
            shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
            pair_ij_mapping0, q_cond_ij0, s_cond_ij0 = bas_pair_cache[i,j]
            pair_kl_mapping0, q_cond_kl0, s_cond_kl0 = bas_pair_cache[k,l]
            if pair_ij_mapping0.size == 0 or pair_kl_mapping0.size == 0:
                continue
            ish_ij = pair_ij_mapping0 // mol.nbas
            ish_kl = pair_kl_mapping0 // mol.nbas
            llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
            scheme = schemes[task]
            component_groups = ((0, 1), (1, 0)) if group1_size is not None else ((None, None),)
            for comp_ij, comp_kl in component_groups:
                if comp_ij is None:
                    pair_ij_mapping = pair_ij_mapping0
                    q_cond_ij = q_cond_ij0
                    s_cond_ij = s_cond_ij0
                else:
                    # Reuse the selected shell pairs for both screening bounds.
                    pair_idx = cupy.where(_shell_group[ish_ij] == comp_ij)[0]
                    pair_ij_mapping = pair_ij_mapping0[pair_idx]
                    q_cond_ij = q_cond_ij0[pair_idx]
                    s_cond_ij = s_cond_ij0[pair_idx]

                if comp_kl is None:
                    pair_kl_mapping = pair_kl_mapping0
                    q_cond_kl = q_cond_kl0
                    s_cond_kl = s_cond_kl0
                else:
                    pair_idx = cupy.where(_shell_group[ish_kl] == comp_kl)[0]
                    pair_kl_mapping = pair_kl_mapping0[pair_idx]
                    q_cond_kl = q_cond_kl0[pair_idx]
                    s_cond_kl = s_cond_kl0[pair_idx]
                npairs_ij = pair_ij_mapping.size
                npairs_kl = pair_kl_mapping.size
                if npairs_ij == 0 or npairs_kl == 0:
                    continue
                err = kern(
                    ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dm_ptrs.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_shell_component.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_local_ao_loc.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_component_nao.data.ptr, ctypes.c_void_p),
                    rys_envs, (ctypes.c_int*2)(*scheme),
                    (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(s_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(s_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_diffuse_exps.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff),
                    ctypes.c_float(dm_penalty),
                    ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dd_pool.data.ptr, ctypes.c_void_p),
                    mol._atm.ctypes, ctypes.c_int(mol.natm),
                    mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
                if err != 0:
                    raise RuntimeError(f'RYS_per_atom_jk_ip1 kernel for {llll} failed')
                kern_counts += 1
                if log.verbose >= logger.DEBUG1:
                    ntasks = npairs_ij * npairs_kl
                    msg = f'processing {llll} on Device {device_id} tasks ~= {ntasks}'
                    t1 = timing_collection.collect(llll, t1, msg)
        return ejk, kern_counts, timing_collection

    results = rhf_grad.multi_gpu.run(proc, non_blocking=True)
    ejk = rhf_grad.multi_gpu.array_reduce([x[0] for x in results], inplace=True)

    if log.verbose >= logger.DEBUG1:
        log.debug1('kernel launches %d', sum(x[1] for x in results))
        rhf_grad._TimingCollector.summary(log.debug1, (x[2] for x in results))

    log.timer_debug1('grad j energy', *cput0)
    return ejk.get()


def _grad_eri_group(mf_grad, dms, keys1, keys2, atmlst):
    '''Evaluate all Coulomb-gradient edges between two component groups.'''
    from gpu4pyscf.neo import int3c2e_bdiv

    mf = mf_grad.base
    mol1 = mf.components[keys1[0]].mol
    for t in keys1[1:]:
        mol1 = mol1 + mf.components[t].mol
    if keys2 is None:
        mol = mol1
        keys = keys1
        group1_size = None
    else:
        mol2 = mf.components[keys2[0]].mol
        for t in keys2[1:]:
            mol2 = mol2 + mf.components[t].mol
        mol = mol1 + mol2
        keys = keys1 + keys2
        group1_size = len(keys1)
    atom_component = numpy.hstack([
        numpy.full(mf.components[t].mol.natm, i, dtype=numpy.int32)
        for i, t in enumerate(keys)])

    vhfopt = rhf_grad._VHFOpt(mol, mf.direct_scf_tol)
    # Copied from scf.jk._VHFOpt.build.
    log = logger.new_logger(vhfopt.mol)
    cput0 = log.init_timer()
    mol = vhfopt.sorted_mol = SortedGTO.from_mol(
        vhfopt.mol, decontract=True, diffuse_cutoff=0.3)
    l_ctr_counts = mol.l_ctr_counts

    # very high angular momentum basis are processed on CPU
    lmax = mol.uniq_l_ctr[:,0].max()
    nbas_by_l = [l_ctr_counts[mol.uniq_l_ctr[:,0]==l].sum() for l in range(lmax+1)]
    l_slices = numpy.append(0, numpy.cumsum(nbas_by_l))
    if lmax > rhf_grad.LMAX:
        vhfopt.h_shls = l_slices[rhf_grad.LMAX+1:].tolist()
    else:
        vhfopt.h_shls = []

    # NEO: remove cross-component shell pairs before overlap and Schwarz
    # screening constructs the exact-gradient work list.
    shell_component = numpy.asarray(atom_component)[mol._bas[:,ATOM_OF]]
    vhfopt.bas_pair_cache = int3c2e_bdiv._cache_q_cond_and_non0pairs(
        mol, vhfopt.rys_envs, vhfopt.direct_scf_tol, shell_component,
        tile=vhfopt.tile)
    log.timer('Initialize q_cond', *cput0)
    # End copied block.
    de = _j_intercomponent_energy_per_atom(
        vhfopt, [mf.components[t].mol for t in keys],
        [mf.components[t].charge*dms[t] for t in keys], group1_size)
    de = de.reshape(len(keys), mf.mol.natm, 3).sum(axis=0)
    return de[list(atmlst)]


def grad_eri(mf_grad, dm0, interactions, atmlst):
    mf = mf_grad.base
    interactions = list(interactions)
    component_keys = [t for t in mf.components
                      if any(t in pair for pair, _ in interactions)]
    unrestricted = {}
    for (t1, t2), interaction in interactions:
        unrestricted[t1] = interaction.mf1_unrestricted
        unrestricted[t2] = interaction.mf2_unrestricted
    dms = {}
    for t in component_keys:
        dm = cupy.asarray(dm0[t])
        if unrestricted[t]:
            assert dm.ndim > 2 and dm.shape[0] == 2
            dm = dm[0] + dm[1]
        dms[t] = dm

    de = numpy.zeros((len(atmlst), 3))
    # Component IDs remove nuclear self-interactions from the grouped N-N job.
    nuclear_keys = [t for t in component_keys if t != 'e']
    if 'e' in component_keys and nuclear_keys:
        de += _grad_eri_group(mf_grad, dms, ['e'], nuclear_keys, atmlst)
    component_keys = nuclear_keys
    if len(component_keys) > 1:
        de += _grad_eri_group(mf_grad, dms, component_keys, None, atmlst)
    return de


def grad_int(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None: mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None:
        atmlst = range(mol.natm)

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    de = grad_eri(mf_grad, dm0, mf.interactions.items(), atmlst)

    if mf_grad.verbose >= logger.DEBUG:
        log = logger.Logger(mf_grad.stdout, mf_grad.verbose)
        log.debug('gradients of Coulomb interaction')
        rhf_grad_cpu._write(log, mol, de, atmlst)
    return de


def _grouped_hcore_energy(components, dm0, dme0):
    from gpu4pyscf.df.int3c2e_bdiv import int3c2e_scheme
    from gpu4pyscf.neo import int1e, int3c2e_bdiv

    mols = {t: comp.mol for t, comp in components.items()}
    mol = next(iter(mols.values()))
    # Follow rhf._grad_nuc_without_ecp with one optimizer whose AO-pair list
    # contains the component-local pairs from all nuclear components.
    auxmol = gto.fakemol_for_charges(mol.atom_coords())
    int3c2e_opt = int3c2e_bdiv.Int3c2eOpt(mols, auxmol).build()
    component_mols, local_ao_loc = int3c2e_opt._component_mol_data()

    dms = {t: component_mols[t].apply_C_mat_CT(dm0[t]) for t in components}
    # Each target nucleus sees the classical nuclear potential scaled by its
    # own charge.
    auxvec = {t: cupy.asarray(-mols[t].atom_charges() * components[t].base.charge,
                              dtype=numpy.float64) for t in components}

    combined_mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
        short_range=False, gout_width=54, deriv=(1,0,0))
    lmax = combined_mol.uniq_l_ctr[:,0].max()
    laux = auxmol.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
    bas_ij_idx, shl_pair_offsets = int3c2e_bdiv._aggregate_shl_pair_blocks(
        combined_mol, int3c2e_opt.bas_ij_blocks, nsp_per_block[0]*16)
    ksh_offsets = numpy.append(0, numpy.cumsum(auxmol.l_ctr_counts))
    ksh_offsets_gpu = cupy.asarray(ksh_offsets + combined_mol.nbas, dtype=numpy.int32)

    # rhf._grad_nuc_without_ecp passes one dense DM and one charge vector.  The
    # grouped kernel selects the corresponding component-local pair here.
    component_index = {t: i for i, t in enumerate(components)}
    pair_component = cupy.asarray(numpy.hstack([
        numpy.full(len(bas_ij), component_index[t], dtype=numpy.int32)
        for (_, bas_ij), t in zip(
            int3c2e_opt.bas_ij_blocks, int3c2e_opt.block_components)]))
    dm_ptrs = cupy.asarray(numpy.asarray(
        [dms[t].data.ptr for t in components], dtype=numpy.uintp))
    auxvec_ptrs = cupy.asarray(numpy.asarray(
        [auxvec[t].data.ptr for t in components], dtype=numpy.uintp))
    local_ao_loc = cupy.asarray(local_ao_loc, dtype=numpy.int32)
    component_nao = cupy.asarray([component_mols[t].nao for t in components], dtype=numpy.int32)

    de = cupy.zeros((combined_mol.natm, 3))
    de_aux = cupy.zeros_like(de)
    err = rhf_grad.libvhf_rys.sum_j_int3c2e_ip1_multi_in(
        ctypes.cast(de.data.ptr, ctypes.c_void_p),
        ctypes.cast(de_aux.data.ptr, ctypes.c_void_p),
        ctypes.cast(dm_ptrs.data.ptr, ctypes.c_void_p),
        ctypes.cast(auxvec_ptrs.data.ptr, ctypes.c_void_p),
        ctypes.byref(int3c2e_opt.int3c2e_envs),
        ctypes.c_int(shm_size_max),
        ctypes.c_int(len(shl_pair_offsets) - 1),
        ctypes.c_int(len(ksh_offsets) - 1),
        ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(ksh_offsets_gpu.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
        ctypes.cast(pair_component.data.ptr, ctypes.c_void_p),
        ctypes.cast(local_ao_loc.data.ptr, ctypes.c_void_p),
        ctypes.cast(component_nao.data.ptr, ctypes.c_void_p),
        ctypes.c_int(combined_mol.natm))
    if err != 0:
        raise RuntimeError('int3c2e_ejk_ip1 failed')

    natm = mol.natm
    attraction = de_aux[:natm]
    # AO derivatives are indexed by the repeated atoms in the concatenated
    # molecule; auxiliary-center derivatives use the single physical atom set.
    p0 = 0
    for t in components:
        p1 = p0 + mols[t].natm
        attraction += de[p0:p1]
        p0 = p1
    attraction *= 2

    int1e_opt = int1e.Int1eOpt(mols, hermi=0)
    ipkin = int1e_opt.get_ipkin()
    ipovlp = int1e_opt.get_ipovlp()
    de = attraction.get()
    for t, comp in components.items():
        # The remaining kinetic, overlap, and extra-force contractions retain
        # the component gradient formula; only their integral builds are shared.
        dh = -rhf_grad.contract_h1e_dm(comp.mol, ipkin[t] / comp.base.mass, dm0[t], hermi=1)
        dh += rhf_grad.contract_h1e_dm(comp.mol, ipovlp[t], dme0[t], hermi=1)
        de += dh
    return de


def grad_epc(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    mf = mf_grad.base
    mol = mf_grad.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    if not hasattr(mf, 'epc') or mf.epc is None:
        return numpy.zeros((len(atmlst), 3))
    de = cupy.zeros((len(atmlst), 3))

    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None: mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    mf_e = mf.components['e']
    mol_e = mf_e.mol
    dm_e = cupy.asarray(dm0['e'])
    if isinstance(mf_e, scf.uhf.UHF):
        assert dm_e.ndim > 2 and dm_e.shape[0] == 2
        dm_e = dm_e[0] + dm_e[1]

    grids_e = mf_e.grids
    if grids_e.coords is None:
        rks.initialize_grids(mf_e, mol_e, dm_e)
    elec_grids_hash = ks._hash_grids(grids_e)
    grids_changed = (mf._elec_grids_hash != elec_grids_hash)
    if grids_changed and mf._epc_n_types is not None:
        if len(mf._epc_n_types) > 0:
            mf._skip_epc = False
    if getattr(mf, '_skip_epc', False):
        return de.get()

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
        return de.get()

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
            return de.get()
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
    dm_e_sorted = opt_e.sort_orbitals(dm_e, axis=[0,1])

    opt_n = ni_n.gdftopt
    sorted_mol_n = opt_n._sorted_mol
    ao_idx_n = cupy.asarray(opt_n._ao_idx)
    grids_n = copy.copy(grids)
    grids_n.mol = mol_n_all
    grids_n._non0ao_idx = None
    non0ao_idx_n = grids_n.get_non0ao_idx(opt_n)

    dm_n_all = cupy.zeros((mol_n_all.nao, mol_n_all.nao))
    for n_type in n_types:
        n0, n1 = n_slices[n_type]
        dm_n_all[n0:n1,n0:n1] = cupy.asarray(dm0[n_type])
    dm_n_all = opt_n.sort_orbitals(dm_n_all, axis=[0,1])

    nao_e = sorted_mol_e.nao
    nao_n = sorted_mol_n.nao
    vxc_e = cupy.zeros((3, nao_e, nao_e))
    vxc_n = cupy.zeros((3, nao_n, nao_n))

    p1 = 0
    for ao_e, idx_e, weight, coords in ni_e.block_loop(
            sorted_mol_e, grids, nao_e, deriv=1):
        p0, p1 = p1, p1 + weight.size
        if len(idx_e) == 0:
            continue
        dm_e_mask = dm_e_sorted[idx_e[:,None],idx_e]
        rho_e = numint.eval_rho(sorted_mol_e, ao_e[0], dm_e_mask, hermi=1)
        rho_e = cupy.maximum(rho_e, 0)
        common = ks.precompute_epc_electron(mf.epc, rho_e)

        vxc_e_grid = 0
        block_id = p0 // numint.MIN_BLK_SIZE
        pad, idx_n, non0shl_idx, ctr_offsets_slice, ao_loc_slice = non0ao_idx_n[block_id]
        if len(idx_n) == 0:
            continue

        ao_n = numint.eval_ao(
            sorted_mol_n, coords, deriv=1,
            screening_info=(len(idx_n), non0shl_idx, ao_loc_slice, ctr_offsets_slice),
            gdftopt=opt_n,
            transpose=False)
        if pad > 0:
            ao_n[:,-pad:] = 0.0
        orig_idx_n = ao_idx_n[idx_n]

        # TODO: Group the component-local nuclear density and derivative-matrix
        # contractions without forming a padded block-diagonal DM.
        for n_type in n_types:
            n0, n1 = n_slices[n_type]
            mask_n = (orig_idx_n >= n0) & (orig_idx_n < n1)
            # Share component AO selection between density and AO derivatives.
            idx = cupy.where(mask_n)[0]
            if idx.size == 0:
                continue
            idx_n_t = idx_n[idx]
            ao_n_t = ao_n[:,idx]
            dm_n_mask = dm_n_all[idx_n_t[:,None],idx_n_t]
            rho_n = numint.eval_rho(sorted_mol_n, ao_n_t[0], dm_n_mask, hermi=1)
            rho_n = cupy.maximum(rho_n, 0)
            _, vxc_n_grid, vxc_e_grid_t = ks.eval_epc(common, rho_n)
            vxc_e_grid += vxc_e_grid_t

            aow_n = numint._scale_ao(ao_n_t[0], weight * vxc_n_grid)
            vtmp = rks_grad._d1_dot_(ao_n_t[1:4], aow_n.T)
            add_sparse(vxc_n, vtmp, idx_n_t)

        aow_e = numint._scale_ao(ao_e[0], weight * vxc_e_grid)
        vtmp = rks_grad._d1_dot_(ao_e[1:4], aow_e.T)
        add_sparse(vxc_e, vtmp, idx_e)

    vxc_e = opt_e.unsort_orbitals(vxc_e, axis=[1,2])
    vxc_n = opt_n.unsort_orbitals(vxc_n, axis=[1,2])

    aoslices = mol_e.aoslice_by_atom()
    for i0, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]
        de[i0] -= cupy.einsum('xij,ij->x', vxc_e[:,p0:p1], dm_e[p0:p1]).real * 2

    for n_type in n_types:
        n0, n1 = n_slices[n_type]
        dm_n = cupy.asarray(dm0[n_type])
        vxc_n_t = vxc_n[:,n0:n1,n0:n1]
        aoslices = mf.components[n_type].mol.aoslice_by_atom()
        for i0, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia,2:]
            if p1 > p0:
                de[i0] -= cupy.einsum('xij,ij->x', vxc_n_t[:,p0:p1], dm_n[p0:p1]).real * 2
    return de.get()


class Gradients(rhf_grad.GradientsBase):
    def __init__(self, mf):
        super().__init__(mf)
        self.grid_response = None
        self.components = {}
        for t, comp in self.base.components.items():
            self.components[t] = general_grad(comp.nuc_grad_method())
        self._keys = self._keys.union(['grid_response', 'components'])

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self.base.reset(self.mol)
        if (self.components.keys() == self.base.components.keys() and
            all(comp.base is self.base.components[t] for t, comp in self.components.items())):
            for t, comp in self.components.items():
                comp.reset(self.mol.components[t])
        else:
            self.components.clear()
            for t, comp in self.base.components.items():
                self.components[t] = general_grad(comp.nuc_grad_method())
        return self

    def grad_nuc(self, mol=None, atmlst=None):
        if mol is None: mol = self.mol
        g_qm = self.components['e'].grad_nuc(mol.components['e'], atmlst)
        if mol.mm_mol is None:
            return g_qm

        mm_mol = mol.mm_mol
        coords = mm_mol.atom_coords()
        charges = mm_mol.atom_charges()
        if mm_mol.charge_model == 'gaussian':
            expnts = mm_mol.get_zetas()
            radii = 1 / numpy.sqrt(expnts)

        mol_e = mol.components['e']
        g_mm = numpy.empty((mol_e.natm, 3))
        for i in range(mol_e.natm):
            q1 = mol_e.atom_charge(i)
            r1 = mol_e.atom_coord(i)
            r = lib.norm(r1 - coords, axis=1)
            if mm_mol.charge_model != 'gaussian':
                coulkern = 1 / r**3
            else:
                coulkern = erf(r/radii)/r - 2/(numpy.sqrt(numpy.pi)*radii) * numpy.exp(-expnts*r**2)
                coulkern = coulkern / r**2
            g_mm[i] = -q1 * numpy.einsum('i,ix,i->x', charges, r1-coords, coulkern)
        if atmlst is not None:
            g_mm = g_mm[atmlst]
        return g_qm + g_mm

    def symmetrize(self, de, atmlst=None):
        return rhf_grad_cpu.symmetrize(self.mol.components['e'], de, atmlst)

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        if mo_energy is None:
            if self.base.mo_energy is None:
                self.base.run()
            mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        if self.grid_response and hasattr(self.base, 'epc') and self.base.epc is not None:
            raise NotImplementedError('Grid response for NEO EPC gradients')

        de = 0
        grouped_components = {}
        for t, comp in self.components.items():
            if self.grid_response is not None and hasattr(comp, 'grid_response'):
                comp.grid_response = self.grid_response
            if t.startswith('n'):
                grouped_components[t] = comp
            else:
                de += comp.grad_elec(mo_energy=mo_energy[t],
                                     mo_coeff=mo_coeff[t],
                                     mo_occ=mo_occ[t])
        if grouped_components:
            dm0 = {}
            dme0 = {}
            for t, comp in grouped_components.items():
                dm0[t] = comp.base.make_rdm1(mo_coeff[t], mo_occ[t])
                dme0[t] = comp.make_rdm1e(mo_energy[t], mo_coeff[t], mo_occ[t])
            de += _grouped_hcore_energy(grouped_components, dm0, dme0)
            for comp in grouped_components.values():
                de += cupy.asnumpy(comp.extra_force())

        de += self.grad_int(mo_energy, mo_coeff, mo_occ)
        if hasattr(self.base, 'epc') and self.base.epc is not None:
            de += self.grad_epc(mo_energy, mo_coeff, mo_occ)

        self.de = de + self.grad_nuc()
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de)
        if self.base.do_disp():
            self.de += self.components['e'].get_dispersion()
        if atmlst is not None:
            self.de = self.de[atmlst]
        logger.timer(self, 'CNEO gradients', *cput0)
        self._finalize()
        return self.de

    grad = lib.alias(kernel, alias_name='grad')

    grad_int = grad_int
    grad_epc = grad_epc
    as_scanner = rhf_grad.as_scanner
    to_cpu = utils.to_cpu
    to_gpu = utils.to_gpu
    device = utils.device

    def get_jk(self, mol=None, dm=None, hermi=0, omega=None):
        raise AttributeError

    def get_j(self, mol=None, dm=None, hermi=0, omega=None):
        raise AttributeError

    def get_k(self, mol=None, dm=None, hermi=0, omega=None):
        raise AttributeError

    def grad_hcore_mm(self, dm=None, mol=None):
        if mol is None: mol = self.mol
        mm_mol = mol.mm_mol
        if mm_mol is None:
            warnings.warn('Not a QM/MM calculation, grad_mm should not be called!')
            return None
        if dm is None: dm = self.base.make_rdm1()

        coords = mm_mol.atom_coords()
        charges = mm_mol.atom_charges()
        expnts = mm_mol.get_zetas()
        g = cupy.zeros(coords.shape)
        # TODO: Contract the MM potential derivatives for all component-local
        # DMs in one launch without forming a padded block-diagonal DM.
        for t, comp in self.base.components.items():
            dm_comp = cupy.asarray(dm[t])
            if dm_comp.ndim > 2:
                dm_comp = dm_comp[0] + dm_comp[1]
            g += int1e_grids_ip2(comp.mol, coords, dm=dm_comp,
                                 charge_exponents=expnts).T * comp.charge
        return (g * cupy.asarray(charges[:,None])).get()

    contract_hcore_mm = grad_hcore_mm

    def grad_nuc_mm(self, mol=None):
        if mol is None: mol = self.mol
        mm_mol = mol.mm_mol
        if mm_mol is None:
            warnings.warn('Not a QM/MM calculation, grad_mm should not be called!')
            return None
        coords = mm_mol.atom_coords()
        charges = mm_mol.atom_charges()
        if mm_mol.charge_model == 'gaussian':
            expnts = mm_mol.get_zetas()
            radii = 1 / numpy.sqrt(expnts)
        g_mm = numpy.zeros_like(coords)
        mol_e = mol.components['e']
        for i in range(mol_e.natm):
            q1 = mol_e.atom_charge(i)
            r1 = mol_e.atom_coord(i)
            r = lib.norm(r1 - coords, axis=1)
            if mm_mol.charge_model != 'gaussian':
                coulkern = 1 / r**3
            else:
                coulkern = erf(r/radii)/r - 2/(numpy.sqrt(numpy.pi)*radii) * numpy.exp(-expnts*r**2)
                coulkern = coulkern / r**2
            g_mm += q1 * numpy.einsum('i,ix,i->ix', charges, r1-coords, coulkern)
        return g_mm

    def grad_mm(self, dm=None, mol=None):
        return self.grad_hcore_mm(dm, mol) + self.grad_nuc_mm(mol)


Grad = Gradients
