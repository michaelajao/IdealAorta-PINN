import importlib.util
import unittest
import numpy as np
from sparse_backend import sparse_products
from compact_rbf import solve_regularized


@unittest.skipUnless(importlib.util.find_spec('scipy') is not None,'scientific sparse environment required')
class ProductBackendTests(unittest.TestCase):
    def matrix(self):
        from scipy.sparse import csr_matrix
        rng=np.random.default_rng(238)
        a=rng.normal(size=(80,12))*np.geomspace(.03,3,12)
        a[rng.random(a.shape)<.5]=0
        return csr_matrix(a),rng.normal(size=80)

    def test_cpu_product_contract(self):
        a,b=self.matrix();operator,metadata=sparse_products(a,'cpu')
        np.testing.assert_array_equal(operator@np.ones(a.shape[1]),a@np.ones(a.shape[1]))
        np.testing.assert_array_equal(operator.T@b,a.T@b)
        self.assertEqual(metadata['device'],'cpu')

    def test_scaled_refinement_retains_the_total_iteration_budget(self):
        a,b=self.matrix()
        _,record=solve_regularized(a,b,.003,1e-12,1,True)
        self.assertEqual(record['iterations'],1)
        self.assertEqual(record['iteration_limit'],1)
        self.assertEqual(record['istop'],7)
        self.assertEqual(sum(r['iterations'] for r in record['refinement_rounds']),1)

    def test_scaled_early_stop_is_refined_without_accepting_original_gradient_failure(self):
        from unittest.mock import patch
        from scipy.sparse.linalg import lsmr as real_lsmr
        a,b=self.matrix();calls=[]
        def early_once(operator,rhs,**kwargs):
            calls.append(kwargs)
            if len(calls)==1:
                return (np.zeros(operator.shape[1]),2,1,1.,1.,1.,1.,1.)
            return real_lsmr(operator,rhs,**kwargs)
        with patch('scipy.sparse.linalg.lsmr',side_effect=early_once):
            actual,record=solve_regularized(a,b,.003,1e-10,2000,True)
        dense=a.toarray();expected=np.linalg.solve(dense.T@dense+.003*np.eye(a.shape[1]),dense.T@b)
        np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-7)
        self.assertGreater(len(calls),1)
        self.assertLess(calls[1]['atol'],calls[0]['atol'])
        self.assertLessEqual(record['iterations'],2000)
        self.assertLessEqual(record['original_stationarity_Frobenius_normalized'],record['original_stationarity_tolerance'])

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None,'CUDA verification runs on Brosnan')
    def test_cuda_double_products_and_ridge_solution_match_cpu(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('CUDA device unavailable')
        a,b=self.matrix();operator,metadata=sparse_products(a,'cuda:0')
        x=np.linspace(-1,1,a.shape[1])
        np.testing.assert_allclose(operator@x,a@x,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(operator.T@b,a.T@b,atol=1e-12,rtol=1e-12)
        ridge=.003;dense=a.toarray()
        expected=np.linalg.solve(dense.T@dense+ridge*np.eye(a.shape[1]),dense.T@b)
        for scaled in (False,True):
            actual,record=solve_regularized(a,b,ridge,1e-12,2000,scaled,operator)
            np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=1e-8)
            self.assertIn(record['istop'],(1,2,4,5))
        self.assertEqual(metadata['dtype'],'float64')
        _,record=solve_regularized(a,b,ridge,1e-12,1,False,operator)
        self.assertEqual(record['istop'],7)


if __name__=='__main__':unittest.main()
