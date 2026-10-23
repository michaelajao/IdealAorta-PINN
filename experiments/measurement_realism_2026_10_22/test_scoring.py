import importlib.util
from types import SimpleNamespace
import unittest
import numpy as np


class AbsoluteWallShearTests(unittest.TestCase):
    def test_absolute_errors_match_direct_vector_and_magnitude_rmse_on_same_mask(self):
        from scoring import absolute_wss_scores
        reference=np.array([[3.,4.,0.],[0.,0.,2.],[1.,-2.,2.]])
        predicted=np.array([[2.,2.,1.],[100.,200.,300.],[-1.,1.,1.]])
        mask=np.array([True,False,True])
        truth_mag=np.linalg.norm(reference[mask],axis=1);pred_mag=np.linalg.norm(predicted[mask],axis=1)
        relative={'wss_mag_rel_l2':np.linalg.norm(pred_mag-truth_mag)/np.linalg.norm(truth_mag),
                  'wss_vec_rel_l2':np.linalg.norm(predicted[mask]-reference[mask])/np.linalg.norm(reference[mask])}
        actual=absolute_wss_scores(relative,reference,mask)
        self.assertAlmostEqual(actual['wss_mag_rms_Pa'],float(np.sqrt(np.mean((pred_mag-truth_mag)**2))))
        self.assertAlmostEqual(actual['wss_vec_rms_Pa'],float(np.sqrt(np.mean(np.sum((predicted[mask]-reference[mask])**2,axis=1)))))
        self.assertEqual(actual['n_scored_wall_nodes'],2)
        with self.assertRaises(ValueError):absolute_wss_scores(relative,reference,np.zeros(3,bool))


@unittest.skipUnless(importlib.util.find_spec('scipy') is not None and
                     importlib.util.find_spec('idealaorta_pinn') is not None,
                     'historical pressure solver equivalence verified on Brosnan')
class PressureAuditTests(unittest.TestCase):
    def test_audited_solver_matches_historical_and_reports_nonconvergence(self):
        from historical import api_module
        from scoring import make_audited_integrator
        xyz=np.stack(np.meshgrid(np.linspace(.002,.134,6),np.linspace(-.009,.009,6),
                                 np.linspace(-.009,.009,6),indexing='ij'),axis=-1).reshape(-1,3)
        wall=np.array([[0,-.01,-.01],[0,.01,.01],[.135,-.01,.01],[.135,.01,-.01]])
        P=SimpleNamespace(X=xyz,wall_xyz=wall,dwall=np.full(len(xyz),.002))
        historical=api_module('postprocess').PressureIntegrator(xyz,wall,P.dwall)
        audited=make_audited_integrator(P)
        gradient=np.tile([2.,3.,-1.],(len(xyz),1))
        a=historical.integrate(gradient);b=audited.integrate(gradient)
        np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)
        expected=xyz@np.array([2.,3.,-1.]);expected-=expected[audited.o_idx].mean()
        np.testing.assert_allclose(b,expected,atol=2e-7)
        self.assertTrue(audited.solve_records[-1]['converged'])
        audited.integrate(np.random.default_rng(40).normal(size=gradient.shape),iters=1)
        self.assertEqual(audited.solve_records[-1]['istop'],7)
        self.assertFalse(audited.solve_records[-1]['converged'])


if __name__=='__main__':unittest.main()
