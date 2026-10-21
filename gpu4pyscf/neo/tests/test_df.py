import unittest

import cupy
import numpy
from pyscf import neo
import gpu4pyscf.neo as gpu_neo
from gpu4pyscf import scf as gpu_scf


def setUpModule():
    global mol
    mol = neo.M(atom='H 0 0 0; F 0 0 0.9', basis='sto-3g',
                nuc_basis='pb4d', quantum_nuc=[0],
                verbose=0, output='/dev/null')


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class KnownValues(unittest.TestCase):
    def test_cderi_host_prefetch(self):
        mol = neo.M(atom='''O 99.8140 100.8350 101.2320
                            H 99.3292  99.9768 101.0630
                            H 99.1516 101.5610 101.4140
                            O 98.8040  98.5122  97.7581
                            H 99.7821  98.6469  97.9167
                            H 98.4218  99.3265  97.3213''',
                    basis='def2-svp', nuc_basis='pb4d', quantum_nuc=['H'], verbose=0)
        for df_nn in (False, True):
            with self.subTest(df_nn=df_nn):
                mf = gpu_neo.CDFT(mol, xc='PBE0', epc=None).density_fit(
                    auxbasis='def2-svp-jkfit', df_ne=True, df_nn=df_nn,
                    df_ne_j_engine='cderi')
                ref = gpu_neo.CDFT(mol, xc='PBE0', epc=None).density_fit(
                    auxbasis='def2-svp-jkfit', df_ne=True, df_nn=df_nn,
                    df_ne_j_engine='cderi')
                mf.with_df.use_gpu_memory = False
                mf.components['e'].with_df.use_gpu_memory = False
                ref.with_df.use_gpu_memory = True
                ref.components['e'].with_df.use_gpu_memory = True
                dm = mf.get_init_guess()
                expected = ref.get_j(dm=dm)
                dm = {t: cupy.repeat(cupy.asarray(d)[None], 256, axis=0)
                      for t, d in dm.items()}
                mf.get_j(dm=dm)
                for blocks in mf.with_df._cderi.values():
                    self.assertIsInstance(blocks[0], numpy.ndarray)
                cupy.cuda.get_current_stream().synchronize()
                # Queue contractions so successive calls reuse live scratch buffers.
                for trial in range(30):
                    values = [mf.get_j(dm=dm) for i in range(7)]
                    for t in dm:
                        error = cupy.max(cupy.stack([
                            cupy.max(abs(value[t]-expected[t])) for value in values]))
                        self.assertLess(float(error), 2e-8)

    def test_scanner_spin(self):
        mol = neo.M(atom='H 0 0 0; Li 0 0 1.6', basis='sto-3g',
                    nuc_basis='pb4d', quantum_nuc=[0], verbose=0)
        mol2 = neo.M(atom='H 0 0 0; Li 0 0 1.6', basis='sto-3g',
                     nuc_basis='pb4d', quantum_nuc=[0], charge=1, spin=1, verbose=0)
        for df_ne in (None, False, True):
            with self.subTest(df_ne=df_ne):
                mf = gpu_neo.CDFT(mol, xc='PBE0')
                if df_ne is not None:
                    mf = mf.density_fit(auxbasis='weigend', df_ne=df_ne)
                mf.conv_tol = 1e-10
                scanner = mf.nuc_grad_method().as_scanner()
                scanner(mol)
                for mol_test, unrestricted in ((mol2, False), (mol, False), (mol, True)):
                    scanner.base.unrestricted = unrestricted
                    mf_ref = gpu_neo.CDFT(mol_test, xc='PBE0', unrestricted=unrestricted)
                    if df_ne is not None:
                        mf_ref = mf_ref.density_fit(auxbasis='weigend', df_ne=df_ne)
                    mf_ref.conv_tol = 1e-10
                    e, grad = scanner(mol_test)
                    self.assertAlmostEqual(e, mf_ref.scf(), 8)
                    self.assertTrue(abs(grad-mf_ref.nuc_grad_method().kernel()).max() < 1e-6)
                    self.assertEqual(isinstance(scanner.base.components['e'], gpu_scf.uhf.UHF),
                                     unrestricted or mol_test.spin != 0)

    def test_scanner_different_mol(self):
        mol_h2o = neo.M(atom='''O  0.000000  0.000000  0.000000
                                H  0.000000 -0.757000  0.587000
                                H  0.000000  0.757000  0.587000''',
                         basis='sto-3g', nuc_basis='pb4d', quantum_nuc=[1,2],
                         verbose=0)
        for df_ne, df_nn in ((False, False), (True, False), (True, True)):
            with self.subTest(df_ne=df_ne, df_nn=df_nn):
                mf = gpu_neo.CDFT(mol, xc='LDA,VWN', epc=None).density_fit(
                    auxbasis='weigend', df_ne=df_ne, df_nn=df_nn)
                mf.conv_tol = 1e-10
                scanner = mf.nuc_grad_method().as_scanner()
                scanner(mol)
                mf_e = scanner.base.components['e']
                for mol_test in (mol_h2o, mol):
                    mf_ref = gpu_neo.CDFT(mol_test, xc='LDA,VWN', epc=None).density_fit(
                        auxbasis='weigend', df_ne=df_ne, df_nn=df_nn)
                    mf_ref.conv_tol = mf.conv_tol
                    e, grad = scanner(mol_test)
                    self.assertAlmostEqual(e, mf_ref.scf(), 8)
                    numpy.testing.assert_allclose(
                        grad, mf_ref.nuc_grad_method().kernel(), atol=1e-6)
                    self.assertIs(scanner.base.components['e'], mf_e)

    def test_hf_df_ne(self):
        mf_cpu = neo.HF(mol).density_fit(df_ne=True)
        mf_cpu.conv_tol = 1e-10
        e_cpu = mf_cpu.kernel()

        for engine in ('direct', 'cderi'):
            with self.subTest(engine=engine):
                mf_gpu = gpu_neo.HF(mol).density_fit(
                    df_ne=True, df_ne_j_engine=engine)
                mf_gpu.conv_tol = mf_cpu.conv_tol
                e_gpu = mf_gpu.kernel()

                self.assertTrue(mf_gpu.converged)
                self.assertAlmostEqual(e_gpu, e_cpu, 9)

    def test_j_with_dm_sets(self):
        dm = gpu_neo.HF(mol).get_init_guess()
        dm_sets = {t: cupy.stack((cupy.asarray(dm_t), cupy.asarray(dm_t) * .5))
                   for t, dm_t in dm.items()}

        for engine in ('cderi', 'direct'):
            mf = gpu_neo.HF(mol).density_fit(df_ne=True,
                                             df_ne_j_engine=engine)
            vj_sets = mf.with_df.get_jk(dm_sets, with_k=False)[0]
            vj0 = mf.with_df.get_jk(dm, with_k=False)[0]
            dm_half = {t: cupy.asarray(dm_t) * .5 for t, dm_t in dm.items()}
            vj1 = mf.with_df.get_jk(dm_half, with_k=False)[0]
            for t in dm:
                numpy.testing.assert_allclose(vj_sets[t][0].get(),
                                              vj0[t].get(), atol=1e-9)
                numpy.testing.assert_allclose(vj_sets[t][1].get(),
                                              vj1[t].get(), atol=1e-9)

    def test_df_nn(self):
        mol_h2o = neo.M(atom='''O  0.000000  0.000000  0.000000
                                H  0.000000 -0.757000  0.587000
                                H  0.000000  0.757000  0.587000''',
                        basis='sto-3g', nuc_basis='pb4d', quantum_nuc=['H'], verbose=0)
        mf_direct = gpu_neo.HF(mol_h2o).density_fit(
            auxbasis='weigend')
        mf_cderi = gpu_neo.HF(mol_h2o).density_fit(
            auxbasis='weigend', df_ne=True, df_nn=True,
            df_ne_j_engine='cderi')
        mf_no_nn = gpu_neo.HF(mol_h2o).density_fit(
            auxbasis='weigend', df_ne=True, df_nn=False)
        dm = mf_direct.get_init_guess()
        vj_direct = mf_direct.with_df.get_jk(dm, with_k=False)[0]
        vj_cderi = mf_cderi.with_df.get_jk(dm, with_k=False)[0]
        vj_no_nn = mf_no_nn.with_df.get_jk(dm, with_k=False)[0]
        for t in vj_cderi:
            numpy.testing.assert_allclose(vj_direct[t].get(),
                                          vj_cderi[t].get(), atol=1e-9)
        numpy.testing.assert_allclose(vj_direct['e'].get(),
                                      vj_no_nn['e'].get(), atol=1e-9)
        self.assertGreater(abs(vj_direct['n1'] - vj_no_nn['n1']).max(),
                           1e-6)

        def no_cpu_nn_vint(dm):
            raise AssertionError('df_nn=True should not call exact n-n vint')
        mf_direct._get_nn_vint = no_cpu_nn_vint
        mf_direct.get_veff(dm=dm)

    def test_hf_df_ne_grad(self):
        mol_h2o = neo.M(atom='''O  0.000000  0.000000  0.000000
                                H  0.000000 -0.757000  0.587000
                                H  0.000000  0.757000  0.587000''',
                        basis='6-31g', nuc_basis='pb4d', quantum_nuc=[1], verbose=0)
        mf_cpu = neo.HF(mol_h2o).density_fit(df_ne=True)
        mf_cpu.conv_tol = 1e-10
        mf_cpu.kernel()
        ref = mf_cpu.nuc_grad_method().kernel()

        mf = gpu_neo.HF(mol_h2o).density_fit(df_ne=True)
        mf.conv_tol = mf_cpu.conv_tol
        mf.kernel()
        numpy.testing.assert_allclose(mf.nuc_grad_method().kernel(), ref,
                                      atol=2e-7)

    def test_lda_df_ne_epc_grad(self):
        mf_cpu = neo.CDFT(mol, xc='LDA,VWN', epc='17-2').density_fit(
            auxbasis='weigend', df_ne=True)
        mf_cpu.components['e'].grids.atom_grid = (99, 590)
        mf_cpu.conv_tol = 1e-10
        mf_cpu.max_cycle = 100
        mf_cpu.kernel()
        g_cpu = mf_cpu.nuc_grad_method().kernel()

        mf_gpu = gpu_neo.CDFT(mol, xc='LDA,VWN', epc='17-2').density_fit(
            auxbasis='weigend', df_ne=True)
        mf_gpu.components['e'].grids.atom_grid = (99, 590)
        mf_gpu.conv_tol = mf_cpu.conv_tol
        mf_gpu.max_cycle = mf_cpu.max_cycle
        mf_gpu.kernel()
        g_gpu = mf_gpu.nuc_grad_method().kernel()

        self.assertTrue(mf_cpu.converged)
        self.assertTrue(mf_gpu.converged)
        numpy.testing.assert_allclose(g_gpu, g_cpu, atol=2e-6)

    def test_df_nn_grad(self):
        mol_h2o = neo.M(atom='''O  0.000000  0.000000  0.000000
                                H  0.000000 -0.757000  0.587000
                                H  0.000000  0.757000  0.587000''',
                        basis='sto-3g', nuc_basis='pb4d', quantum_nuc=['H'], verbose=0)
        mf = gpu_neo.HF(mol_h2o).density_fit(df_ne=True, df_nn=True)
        mf.conv_tol = 1e-10
        mf.kernel()
        ref = numpy.array([
            [0.0, 0.0, 0.0329282604],
            [0.0, 0.0026452475, -0.0164641302],
            [0.0, -0.0026452475, -0.0164641302]])
        numpy.testing.assert_allclose(mf.nuc_grad_method().kernel(), ref,
                                      atol=1e-8)

    def test_epc_df_nn_grad(self):
        mol_h2o = neo.M(atom='''O  0.000000  0.000000  0.000000
                                H  0.000000 -0.757000  0.587000
                                H  0.000000  0.757000  0.587000''',
                        basis='sto-3g', nuc_basis='pb4d', quantum_nuc=['H'], verbose=0)
        mf = gpu_neo.CDFT(mol_h2o, xc='LDA,VWN', epc='17-2').density_fit(
            auxbasis='weigend', df_ne=True, df_nn=True)
        mf.components['e'].grids.level = 1
        mf.conv_tol = 1e-10
        mf.max_cycle = 100
        mf.kernel()
        ref = numpy.array([
            [0.0, 0.0, 0.103446904247080],
            [0.0, 0.051770424343938, -0.051752079756173],
            [0.0, -0.051770424343948, -0.051752079756169]])
        self.assertTrue(mf.converged)
        numpy.testing.assert_allclose(mf.nuc_grad_method().kernel(), ref,
                                      atol=1e-8)

    def test_hf_electron_only_df(self):
        mf_cpu = neo.HF(mol).density_fit(df_ne=False)
        mf_cpu.conv_tol = 1e-10
        e_cpu = mf_cpu.kernel()

        mf_gpu = gpu_neo.HF(mol).density_fit(df_ne=False)
        mf_gpu.conv_tol = mf_cpu.conv_tol
        e_gpu = mf_gpu.kernel()

        self.assertTrue(mf_gpu.converged)
        self.assertAlmostEqual(e_gpu, e_cpu, 9)

    def test_hf_electron_only_df_grad(self):
        mf_gpu = gpu_neo.HF(mol).density_fit(df_ne=False)
        mf_gpu.conv_tol = 1e-10
        mf_gpu.kernel()
        ref = numpy.array([
            [0.0, 0.0, 0.0394663975],
            [0.0, 0.0, -0.0394663975]])
        numpy.testing.assert_allclose(mf_gpu.nuc_grad_method().kernel(), ref,
                                      atol=1e-8)

    def test_lda_electron_only_df(self):
        mf_cpu = neo.KS(mol, xc='LDA,VWN', epc=None).density_fit(df_ne=False)
        mf_cpu.conv_tol = 1e-9
        e_cpu = mf_cpu.kernel()

        mf_gpu = gpu_neo.KS(mol, xc='LDA,VWN', epc=None).density_fit(df_ne=False)
        mf_gpu.conv_tol = mf_cpu.conv_tol
        e_gpu = mf_gpu.kernel()

        self.assertTrue(mf_gpu.converged)
        self.assertAlmostEqual(e_gpu, e_cpu, 8)

    def test_lda_df_ne(self):
        mf_cpu = neo.KS(mol, xc='LDA,VWN', epc=None).density_fit(df_ne=True)
        mf_cpu.conv_tol = 1e-9
        e_cpu = mf_cpu.kernel()

        mf_gpu = gpu_neo.KS(mol, xc='LDA,VWN', epc=None).density_fit(df_ne=True)
        mf_gpu.conv_tol = mf_cpu.conv_tol
        e_gpu = mf_gpu.kernel()

        self.assertTrue(mf_gpu.converged)
        self.assertAlmostEqual(e_gpu, e_cpu, 8)

    def test_lda_df_ne_epc17_2(self):
        mf_gpu = gpu_neo.KS(mol, xc='LDA,VWN', epc='17-2').density_fit(df_ne=True)
        mf_gpu.conv_tol = 1e-9
        e_gpu = mf_gpu.kernel()

        mf_ref = gpu_neo.KS(mol, xc='LDA,VWN', epc='17-2').density_fit(
            df_ne=True, df_ne_j_engine='cderi')
        mf_ref.conv_tol = mf_gpu.conv_tol
        e_ref = mf_ref.kernel()

        self.assertTrue(mf_gpu.converged)
        self.assertTrue(mf_ref.converged)
        self.assertAlmostEqual(e_gpu, e_ref, 9)

    def test_hybrid_df_ne(self):
        mf_cpu = neo.KS(mol, xc='B3LYP', epc=None).density_fit(df_ne=True)
        mf_cpu.components['e'].grids.level = 1
        mf_cpu.conv_tol = 1e-9
        e_cpu = mf_cpu.kernel()

        mf_gpu = gpu_neo.KS(mol, xc='B3LYP', epc=None).density_fit(
            df_ne=True)
        mf_gpu.components['e'].grids.level = 1
        mf_gpu.conv_tol = mf_cpu.conv_tol
        e_gpu = mf_gpu.kernel()

        self.assertTrue(mf_gpu.converged)
        self.assertAlmostEqual(e_gpu, e_cpu, 8)

    def test_rsh_df_ne(self):
        mf_cpu = neo.KS(mol, xc='CAMB3LYP', epc=None).density_fit(df_ne=True)
        mf_cpu.components['e'].grids.level = 1
        mf_cpu.conv_tol = 1e-9
        e_cpu = mf_cpu.kernel()

        mf_gpu = gpu_neo.KS(mol, xc='CAMB3LYP', epc=None).density_fit(df_ne=True)
        mf_gpu.components['e'].grids.level = 1
        mf_gpu.conv_tol = mf_cpu.conv_tol
        e_gpu = mf_gpu.kernel()

        self.assertTrue(mf_gpu.converged)
        self.assertAlmostEqual(e_gpu, e_cpu, 8)

    def test_rsh_mixing_modes(self):
        for spin in (0, 1):
            with self.subTest(spin=spin):
                mol_h2o = neo.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                                basis='6-31g', nuc_basis='pb4d', quantum_nuc=['H'],
                                charge=spin, spin=spin, verbose=0)
                results = []
                for mode in ('mix_outside_kernel', 'mix_inside_kernel'):
                    mf = gpu_neo.CDFT(mol_h2o, xc='CAMB3LYP', epc=None).density_fit(
                        auxbasis='def2-universal-jkfit', df_ne=True, df_nn=True)
                    mf.components['e'].range_separated_mode = mode
                    mf.components['e'].grids.level = 1
                    mf.conv_tol = 1e-11
                    mf.conv_tol_grad = 1e-7
                    energy = mf.kernel()
                    self.assertTrue(mf.converged)
                    grad = mf.nuc_grad_method().kernel()
                    results.append((energy, grad))
                # Fitting the mixed operator differs from mixing separately
                # fitted operators, as in the original RSH integration tests.
                self.assertLess(abs(results[0][0] - results[1][0]), 1e-6)
                numpy.testing.assert_allclose(results[0][1], results[1][1],
                                              atol=1e-6, rtol=0)

    def test_schur_e_cderi_matches_eonly_df(self):
        mf_gpu = gpu_neo.KS(mol, xc='B3LYP', epc=None).density_fit(
            df_ne=True, df_ne_j_engine='cderi')
        dm = mf_gpu.get_init_guess()
        mf_gpu.get_veff(dm=dm)

        cderi = numpy.concatenate([cupy.asnumpy(c)
                                   for c in mf_gpu.components['e'].with_df._cderi])
        ref_df = gpu_scf.HF(mol.components['e']).density_fit(
            auxbasis=mf_gpu.with_df.auxbasis).with_df
        ref_df.build()
        ref = numpy.concatenate([cupy.asnumpy(c) for c in ref_df._cderi])

        self.assertEqual(cderi.shape, ref.shape)
        self.assertLess(numpy.linalg.norm(cderi - ref), 1e-10)

    def test_schur_cpu_memory_e_cderi_is_view(self):
        mf_gpu = gpu_neo.KS(mol, xc='B3LYP', epc=None).density_fit(
            df_ne=True, df_ne_j_engine='cderi')
        mf_gpu.with_df.use_gpu_memory = False
        dm = mf_gpu.get_init_guess()
        mf_gpu.get_veff(dm=dm)

        naux = 0
        for cderi, cderi_e in zip(mf_gpu.with_df._cderi['e'],
                                  mf_gpu.components['e'].with_df._cderi):
            self.assertIsInstance(cderi, numpy.ndarray)
            if cderi_e.size:
                self.assertTrue(numpy.shares_memory(cderi, cderi_e))
            naux += cderi_e.shape[0]
        self.assertEqual(naux, mf_gpu.components['e'].with_df.naux)

    def test_schur_rsh_k_matches_eonly_df(self):
        mf_gpu = gpu_neo.KS(mol, xc='CAMB3LYP', epc=None).density_fit(
            df_ne=True, df_ne_j_engine='cderi')
        dm = mf_gpu.get_init_guess()
        mf_gpu.get_veff(dm=dm)
        omega = mf_gpu.components['e']._numint.rsh_and_hybrid_coeff(
            mf_gpu.components['e'].xc, spin=mol.components['e'].spin)[0]

        vk = mf_gpu.components['e'].with_df.get_jk(
            dm['e'], with_j=False, with_k=True, omega=omega)[1]
        ref_df = gpu_scf.HF(mol.components['e']).density_fit(
            auxbasis=mf_gpu.with_df.auxbasis).with_df
        ref = ref_df.get_jk(dm['e'], with_j=False, with_k=True,
                            omega=omega)[1]

        self.assertLess(cupy.linalg.norm(vk - ref), 1e-9)

if __name__ == '__main__':
    print('Full Tests for GPU NEO density fitting')
    unittest.main()
