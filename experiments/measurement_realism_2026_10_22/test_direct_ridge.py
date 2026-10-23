"""Direct solver checks against independent augmented least-squares solutions."""
import importlib.util
import unittest
import numpy as np
from direct_ridge import DirectRidgeSystem,allocation_plan,GiB


@unittest.skipUnless(importlib.util.find_spec('scipy'),'SciPy required for direct sparse solves')
class DirectRidgeTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'),'Torch CPU factorization requires PyTorch')
    def test_torch_cpu_factor_and_refinement_match_independent_augmented_solution(self):
        from scipy.sparse import csr_matrix
        rng=np.random.default_rng(396);a=rng.normal(size=(71,23));a[np.abs(a)<.8]=0
        b=rng.normal(size=(71,6));ridge=1e-5
        expected=np.linalg.lstsq(np.vstack([a,np.sqrt(ridge)*np.eye(23)]),
                                np.vstack([b,np.zeros((23,6))]),rcond=None)[0]
        system=DirectRidgeSystem(csr_matrix(a),b,cpu_factor_backend='torch',block_columns=7)
        system.gram*=1.0001
        actual,record=system.solve(ridge,stationarity_tolerance=1e-12)
        np.testing.assert_allclose(actual,expected,atol=2e-11,rtol=2e-11)
        self.assertTrue(record['converged']);self.assertGreater(len(record['refinement_rounds']),1)
        self.assertEqual(record['factor_backend'],'Torch cpu Cholesky')
        self.assertEqual(system.plan['cpu_factor_backend'],'torch')
        system.close()

    @unittest.skipUnless(importlib.util.find_spec('torch'),'CUDA equivalence requires PyTorch')
    def test_cuda_block_products_and_both_factor_devices_match_augmented_solution(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        from scipy.sparse import csr_matrix
        rng=np.random.default_rng(390);a=rng.normal(size=(67,19));a[np.abs(a)<.8]=0
        b=rng.normal(size=(67,2));ridge=1e-5
        expected=np.linalg.lstsq(np.vstack([a,np.sqrt(ridge)*np.eye(19)]),
                                np.vstack([b,np.zeros((19,2))]),rcond=None)[0]
        for factor_device,cpu_backend in [('cpu','scipy'),('cpu','torch'),('cuda:0','scipy')]:
            system=DirectRidgeSystem(csr_matrix(a),b,device='cuda:0',factor_device=factor_device,
                                     cpu_factor_backend=cpu_backend,block_columns=7)
            actual,record=system.solve(ridge)
            np.testing.assert_allclose(actual,expected,atol=2e-11,rtol=2e-11)
            self.assertTrue(record['converged'])
            system.close()

    def test_blocked_multi_rhs_matches_augmented_lstsq_and_preserves_inputs(self):
        from scipy.sparse import csr_matrix
        rng=np.random.default_rng(391);a=rng.normal(size=(79,17));a[np.abs(a)<.7]=0
        b=rng.normal(size=(79,6));ridge=.003
        expected=np.linalg.lstsq(np.vstack([a,np.sqrt(ridge)*np.eye(17)]),
                                np.vstack([b,np.zeros((17,6))]),rcond=None)[0]
        matrix=csr_matrix(a);before=matrix.data.copy();target=b.copy()
        for block in (1,7,32):
            system=DirectRidgeSystem(matrix,b,block_columns=block)
            actual,record=system.solve(ridge)
            np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=2e-12)
            self.assertTrue(record['converged'])
            self.assertLess(max(record['original_stationarity_Frobenius_normalized']),1e-12)
            np.testing.assert_array_equal(matrix.data,before);np.testing.assert_array_equal(b,target)
            system.close()

    def test_duplicate_columns_and_small_ridge_remain_audited(self):
        from scipy.sparse import csr_matrix
        rng=np.random.default_rng(392);a=rng.normal(size=(55,9));a[:,8]=a[:,0]
        a*=np.geomspace(.01,10,9);b=rng.normal(size=(55,2));ridge=1e-7
        expected=np.linalg.lstsq(np.vstack([a,np.sqrt(ridge)*np.eye(9)]),
                                np.vstack([b,np.zeros((9,2))]),rcond=None)[0]
        system=DirectRidgeSystem(csr_matrix(a),b,block_columns=4)
        actual,record=system.solve(ridge)
        np.testing.assert_allclose(a@actual,a@expected,atol=2e-8,rtol=2e-8)
        self.assertTrue(record['converged'])
        system.close()

    def test_unacceptable_original_gradient_cannot_be_certified_by_factorization(self):
        from scipy.sparse import csr_matrix
        rng=np.random.default_rng(393);a=rng.normal(size=(30,6));b=rng.normal(size=(30,2))
        system=DirectRidgeSystem(csr_matrix(a),b)
        system.gram*=2 # simulates a corrupted/inaccurate product, but remains SPD
        _,record=system.solve(.01,max_refinement=0)
        self.assertFalse(record['converged'])
        self.assertGreater(max(record['original_stationarity_Frobenius_normalized']),1e-6)
        system.close()

    def test_refinement_corrects_a_small_gram_error_using_original_sparse_gradient(self):
        from scipy.sparse import csr_matrix
        rng=np.random.default_rng(395);a=rng.normal(size=(47,11));b=rng.normal(size=(47,2));ridge=.01
        expected=np.linalg.lstsq(np.vstack([a,np.sqrt(ridge)*np.eye(11)]),
                                np.vstack([b,np.zeros((11,2))]),rcond=None)[0]
        system=DirectRidgeSystem(csr_matrix(a),b);system.gram*=1.0001
        actual,record=system.solve(ridge,stationarity_tolerance=1e-12)
        self.assertTrue(record['converged']);self.assertGreater(len(record['refinement_rounds']),1)
        np.testing.assert_allclose(actual,expected,atol=2e-11,rtol=2e-11)
        system.close()

    def test_memory_gate_precedes_gram_construction_and_bad_inputs_fail(self):
        from scipy.sparse import csr_matrix
        a=csr_matrix(np.eye(5));b=np.ones((5,2))
        with self.assertRaisesRegex(MemoryError,'host allocation'):
            DirectRidgeSystem(a,b,max_host_bytes=1)
        with self.assertRaises(ValueError):DirectRidgeSystem(a,b[:4])
        with self.assertRaises(ValueError):DirectRidgeSystem(a,b,block_columns=0)
        system=DirectRidgeSystem(a,b);system.close()
        with self.assertRaisesRegex(ValueError,'closed'):system.solve(.01)


class AllocationTests(unittest.TestCase):
    def test_vector_design_is_never_included_as_a_dense_gpu_allocation(self):
        shape=(121392,38222);nnz=270000000
        cpu_factor=allocation_plan(shape,nnz,2,512,'cuda:0','cpu')
        gpu_factor=allocation_plan(shape,nnz,2,512,'cuda:0','cuda:0')
        self.assertGreater(cpu_factor['dense_design_bytes_avoided'],30*GiB)
        self.assertLess(cpu_factor['gpu_peak_bound_bytes'],8*GiB)
        self.assertTrue(cpu_factor['host_gram_streaming'])
        self.assertEqual(gpu_factor['gpu_peak_bound_bytes']-cpu_factor['gpu_peak_bound_bytes'],
                         3*cpu_factor['dense_gram_bytes'])


if __name__=='__main__':unittest.main()
