import unittest
import numpy as np
import pyscf
from pyscf import lib
from gpu4pyscf.dft import rks
from gpu4pyscf.qmmm.pbc import itrf

def setUpModule():
    global mol
    atom='''
         O       0.0000000000    -0.0000000000     0.1174000000
         H      -0.7570000000    -0.0000000000    -0.4696000000
         H       0.7570000000     0.0000000000    -0.4696000000
         '''
    mol = pyscf.M(atom=atom, basis='cc-pvdz')

def tearDownModule():
    global mol
    del mol

class KnowValues(unittest.TestCase):
    def test_finite_difference_gradient(self):
        mf = rks.RKS(mol, xc='PBE').density_fit(auxbasis='cc-pvdz-jkfit')
        mf.grids.atom_grid = (99, 974)
        mf.conv_tol = 1e-12
        mf = itrf.add_mm_charges(
            mf,
            [[1,2,-1], [3,4,5]],
            np.eye(3)*15,
            [-5,5],
            [0.8,1.2],
            rcut_ewald=8,
            rcut_hcore=6,
        )
        mf.pop_method = 'meta-lowdin'
        mf.mm_mol.multipole_order = 2
        mf.kernel()
        g = mf.nuc_grad_method().kernel()

        atom1 = '''
            O       0.0000000000    -0.0000000000     0.1175000000
            H      -0.7570000000    -0.0000000000    -0.4696000000
            H       0.7570000000     0.0000000000    -0.4696000000
            '''
        mol1 = pyscf.M(atom=atom1, basis='cc-pvdz')
        mf1 = rks.RKS(mol1, xc='PBE').density_fit(auxbasis='cc-pvdz-jkfit')
        mf1.grids.atom_grid = (99, 974)
        mf1.conv_tol = 1e-12
        mf1 = itrf.add_mm_charges(
            mf1,
            [[1,2,-1], [3,4,5]],
            np.eye(3)*15,
            [-5,5],
            [0.8,1.2],
            rcut_ewald=8,
            rcut_hcore=6,
        )
        mf1.pop_method = 'meta-lowdin'
        mf1.mm_mol.multipole_order = 2
        e1 = mf1.kernel()

        atom2 = '''
                 O       0.0000000000    -0.0000000000     0.1173000000
                 H      -0.7570000000    -0.0000000000    -0.4696000000
                 H       0.7570000000     0.0000000000    -0.4696000000
                 '''
        mol2 = pyscf.M(atom=atom2, basis='cc-pvdz')
        mf2 = rks.RKS(mol2, xc='PBE').density_fit(auxbasis='cc-pvdz-jkfit')
        mf2.grids.atom_grid = (99, 974)
        mf2.conv_tol = 1e-12
        mf2 = itrf.add_mm_charges(
            mf2,
            [[1,2,-1], [3,4,5]],
            np.eye(3)*15,
            [-5,5],
            [0.8,1.2],
            rcut_ewald=8,
            rcut_hcore=6,
        )
        mf2.pop_method = 'meta-lowdin'
        mf2.mm_mol.multipole_order = 2
        e2 = mf2.kernel()

        self.assertAlmostEqual(float(g[0,2]), float(e1-e2)/0.0002*lib.param.BOHR, 5)

    def test_finite_difference_gradient_without_pre_orth(self):
        mf = rks.RKS(mol, xc='PBE').density_fit(auxbasis='cc-pvdz-jkfit')
        mf.grids.atom_grid = (99, 974)
        mf.conv_tol = 1e-12
        mf = itrf.add_mm_charges(
            mf,
            [[1,2,-1], [3,4,5]],
            np.eye(3)*15,
            [-5,5],
            [0.8,1.2],
            rcut_ewald=8,
            rcut_hcore=6,
        )
        mf.pop_method = 'lowdin'
        mf.mm_mol.multipole_order = 2
        mf.pre_orth_ao = None
        mf.kernel()
        g = mf.nuc_grad_method().kernel()

        atom1 = '''
            O       0.0000000000    -0.0000000000     0.1175000000
            H      -0.7570000000    -0.0000000000    -0.4696000000
            H       0.7570000000     0.0000000000    -0.4696000000
            '''
        mol1 = pyscf.M(atom=atom1, basis='cc-pvdz')
        mf1 = rks.RKS(mol1, xc='PBE').density_fit(auxbasis='cc-pvdz-jkfit')
        mf1.grids.atom_grid = (99, 974)
        mf1.conv_tol = 1e-12
        mf1 = itrf.add_mm_charges(
            mf1,
            [[1,2,-1], [3,4,5]],
            np.eye(3)*15,
            [-5,5],
            [0.8,1.2],
            rcut_ewald=8,
            rcut_hcore=6,
        )
        mf1.pop_method = 'lowdin'
        mf1.mm_mol.multipole_order = 2
        mf1.pre_orth_ao = None
        e1 = mf1.kernel()

        atom2 = '''
                 O       0.0000000000    -0.0000000000     0.1173000000
                 H      -0.7570000000    -0.0000000000    -0.4696000000
                 H       0.7570000000     0.0000000000    -0.4696000000
                 '''
        mol2 = pyscf.M(atom=atom2, basis='cc-pvdz')
        mf2 = rks.RKS(mol2, xc='PBE').density_fit(auxbasis='cc-pvdz-jkfit')
        mf2.grids.atom_grid = (99, 974)
        mf2.conv_tol = 1e-12
        mf2 = itrf.add_mm_charges(
            mf2,
            [[1,2,-1], [3,4,5]],
            np.eye(3)*15,
            [-5,5],
            [0.8,1.2],
            rcut_ewald=8,
            rcut_hcore=6,
        )
        mf2.pop_method = 'lowdin'
        mf2.mm_mol.multipole_order = 2
        mf2.pre_orth_ao = None
        e2 = mf2.kernel()

        self.assertAlmostEqual(float(g[0,2]), float(e1-e2)/0.0002*lib.param.BOHR, 5)

    def test(self):
        mf = rks.RKS(mol, xc='PBE').density_fit(auxbasis='cc-pvdz-jkfit')
        mf.grids.atom_grid = (99, 974)
        mf.conv_tol = 1e-12
        mf = itrf.add_mm_charges(
            mf,
            [[1,2,-1], [3,4,5]],
            np.eye(3)*15,
            [-5,5],
            [0.8,1.2],
            rcut_ewald=8,
            rcut_hcore=6,
        )
        mf.pop_method = 'lowdin'
        mf.pre_orth_ao = None
        e = mf.kernel()
        g = mf.nuc_grad_method()
        g_qm = g.kernel()
        g_mm = g.grad_nuc_mm() + g.grad_hcore_mm(mf.make_rdm1()) + g.de_ewald_mm

        e_ref = -76.47335967649721
        g_qm_ref = [[ 0.03102094,  0.08452684, -0.07458882],
                    [-0.01728549, -0.03342796,  0.02045149],
                    [-0.02103994, -0.16630177,  0.05383428]]
        g_mm_ref = [[ 0.00587872,  0.11438000, -0.00052149],
                    [ 0.00142577,  0.00082289,  0.00082455]]

        self.assertAlmostEqual(float(e), e_ref, 8)
        np.testing.assert_array_almost_equal(g_qm, g_qm_ref, 6)
        np.testing.assert_array_almost_equal(g_mm, g_mm_ref, 6)

if __name__ == "__main__":
    print("Full Tests for (meta-)Lowdin Population in QMMM with PBC")
    unittest.main()
