import unittest
import importlib.util
from types import SimpleNamespace
import numpy as np
from metrics import temporal_error_decomposition


class TemporalTests(unittest.TestCase):
    def test_identical_velocity_errors_cancel_in_acceleration(self):
        rng=np.random.default_rng(3)
        truth=rng.normal(size=(2,200,3))
        shared=rng.normal(size=(200,3))*.02
        result=temporal_error_decomposition(truth+shared,truth,[1.775,1.780])
        self.assertLess(result['acceleration_error_rms_vector_m_s2'],1e-12)
        np.testing.assert_allclose(result['cross_time_error_component_correlation'],1,atol=1e-13)

    def test_opposite_errors_amplify_and_identity_includes_bias(self):
        rng=np.random.default_rng(5)
        error=rng.normal(size=(300,3))*.01+np.array([.01,-.02,.03])
        predicted=np.stack([-error,error])
        result=temporal_error_decomposition(predicted,np.zeros_like(predicted),[0,.01])
        expected=float(np.sqrt(np.mean(np.sum((2*error/.01)**2,axis=1))))
        self.assertAlmostEqual(result['acceleration_error_rms_vector_m_s2'],expected,places=12)
        self.assertLess(result['second_moment_identity_max_abs_error_m2_s4'],1e-11)
        np.testing.assert_allclose(result['cross_time_error_component_correlation'],-1,atol=1e-13)
        self.assertAlmostEqual(result['acceleration_error_bias_squared_m2_s4']+
                               result['acceleration_error_variance_sum_m2_s4'],expected**2,places=10)

    def test_invalid_arrays_rejected(self):
        with self.assertRaises(ValueError):temporal_error_decomposition(np.zeros((2,0,3)),np.zeros((2,0,3)),[0,1])
        with self.assertRaises(ValueError):temporal_error_decomposition(np.zeros((2,2,3)),np.zeros((2,2,3)),[1,1])

    @unittest.skipUnless(importlib.util.find_spec('scipy') is not None and
                         importlib.util.find_spec('idealaorta_pinn') is not None,
                         'historical scorer and SciPy are verified on Brosnan')
    def test_cached_predictions_preserve_diagnostics_without_repeating_bulk_inference(self):
        from flux import PlanarCap
        from observations import Observations,common_validation_mask
        from run_pilot import diagnostic_scores
        rng=np.random.default_rng(13);x=rng.uniform(.01,.09,(30,3));times=[0,10]
        def truth(p,t):return p+np.array([.2+t/1000,.1,.05])
        def model(p,t):return truth(p,t)+np.column_stack([.01*p[:,1]+t*.0001,.02*p[:,2],-.01*p[:,0]])
        calls=[]
        def predictor(p,t):
            if np.array_equal(p,x):calls.append(t)
            return model(p,t)
        predictors={t:lambda p,t=t:predictor(p,t) for t in times}
        obs=Observations.make(times,x[:10,None],np.ones(1),np.stack([truth(x[:10],t) for t in times]),
                              common_validation_mask(10,5),np.arange(10),0,9)
        problem=SimpleNamespace(target_ms=10,X=x,sac=np.ones(30,bool),
                                snaps={t:SimpleNamespace(uvw=truth(x,t)) for t in times})
        square=np.array([[0,0,0],[0,.01,0],[0,0,.01],[0,.01,.01]])
        caps=[PlanarCap(square,np.full(4,.000025),np.array([-1,0,0]),0),
              PlanarCap(square+[.1,0,0],np.full(4,.000025),np.array([1,0,0]),0)]
        reference=diagnostic_scores(problem,obs,predictors,np.arange(30),x[:4],caps)
        self.assertEqual(sorted(calls),times)
        calls.clear()
        cached={t:model(x,t) for t in times}
        result=diagnostic_scores(problem,obs,predictors,np.arange(30),x[:4],caps,cached)
        self.assertEqual(calls,[])
        self.assertEqual(result,reference)


if __name__=='__main__':unittest.main()
