"""Numerical/information-contract tests; run all dependency-backed tests on Brosnan."""
import importlib.util
import math
import unittest
from types import SimpleNamespace
import numpy as np
from observations import Observations, common_validation_mask, matched_location_subsets, torch_observe
from operators import voxel_quadrature
from comparators import ReducedRBF, geometry_centres, fit_rbf_pair

HAS_SCIPY = importlib.util.find_spec('scipy') is not None
HAS_TORCH = importlib.util.find_spec('torch') is not None


class InformationTests(unittest.TestCase):
    def setUp(self):
        self.x=np.random.default_rng(1).normal(size=(100,3))
        self.clean=np.stack([self.x,2*self.x])
        self.val=common_validation_mask(100,12)

    def observation(self,sigma=.02):
        return Observations.make([1775,1780],self.x[:,None],np.ones(1),self.clean,
                                 self.val,np.arange(100),sigma,22001)

    def test_immutability_and_exact_common_split(self):
        original=self.clean.copy()
        obs=self.observation()
        self.assertEqual(int(obs.validation.sum()),10)
        self.assertEqual(obs.metadata['n_fit_per_time'],90)
        np.testing.assert_array_equal(self.clean,original)
        with self.assertRaises(ValueError):obs.values[0,0,0]=100
        self.clean[:]=99
        np.testing.assert_array_equal(obs.clean,original)
        with self.assertRaises(ValueError):
            Observations.make([1780,1775],self.x[:,None],np.ones(1),original,self.val,np.arange(100),0,1)

    def test_location_removal_matches_both_strata_and_noise(self):
        obs=self.observation()
        subsets=matched_location_subsets(obs.anchor_ids,np.linspace(0,.004,100),self.val,.00125,32)
        near=obs.subset(subsets['exclude_near_wall'],'near')
        random=obs.subset(subsets['matched_random_removal'],'random')
        self.assertEqual(near.metadata['n_fit_per_time'],random.metadata['n_fit_per_time'])
        self.assertEqual(near.metadata['n_val_per_time'],random.metadata['n_val_per_time'])
        for subset,rows in [(near,subsets['exclude_near_wall']),(random,subsets['matched_random_removal'])]:
            np.testing.assert_array_equal(subset.values,obs.values[:,rows])
            np.testing.assert_array_equal(subset.validation,obs.validation[rows])

    def test_point_volume_noise_is_exactly_paired(self):
        point=self.observation()
        rows=np.arange(10,90)
        support,w=voxel_quadrature(self.x[rows],.2,3)
        average=self.clean[:,rows]+.1
        volume=Observations.make([1775,1780],support,w,average,self.val[rows],rows,.02,22001,
                                realized_noise=(point.values-point.clean)[:,rows])
        np.testing.assert_allclose(volume.values-volume.clean,(point.values-point.clean)[:,rows],atol=5e-16)

    def test_scalar_gaussian_volume_design_matches_exact_integral(self):
        width=.4;ell=.3
        p,w=voxel_quadrature(np.zeros((1,3)),width,7)
        model=ReducedRBF(np.zeros((1,3)),ell,False,np.zeros(3),1)
        integrated=model.design(p,w)[0,0]
        one=ell*math.sqrt(2*math.pi)/width*math.erf(width/(2*math.sqrt(2)*ell))
        self.assertAlmostEqual(integrated,one**3,places=11)
        self.assertLess(integrated,model.design(np.zeros((1,1,3)),np.ones(1))[0,0]-.1)

    def test_solenoidal_affine_extension_is_divergence_free(self):
        rng=np.random.default_rng(21)
        model=ReducedRBF(rng.normal(size=(7,3)),.5,True,np.zeros(3),2)
        model.coefficients=rng.normal(size=model.n_columns)
        query=rng.normal(size=(15,3))
        errors=[]
        for eps in [1e-3,5e-4]:
            div=np.zeros(len(query))
            for c in range(3):
                d=np.eye(3)[c]*eps
                div+=(model.predict(query+d)[:,c]-model.predict(query-d)[:,c])/(2*eps)
            errors.append(np.linalg.norm(div))
        self.assertLess(errors[1],errors[0]*.26)
        self.assertLess(errors[1],1e-4)


@unittest.skipUnless(HAS_SCIPY,'SciPy tested in the Brosnan scientific environment')
class GeometryAndFitTests(unittest.TestCase):
    def test_entire_box_certificate_checks_other_nearest_candidates(self):
        from geometry import OrientedPointDomain
        domain=OrientedPointDomain([[-1,0,0],[1.5,1.5,0]],[[1,0,0],[1,1,0]],margin_m=0)
        self.assertTrue(domain.contains(np.zeros((1,3)))[0])
        self.assertFalse(domain.certify_boxes(np.zeros((1,3)),2)[0])
        halfspace=OrientedPointDomain([[0,0,0]],[[1,0,0]],margin_m=.01)
        np.testing.assert_array_equal(halfspace.certify_boxes([[.6,0,0],[.4,0,0]],1),[True,False])

    def test_operator_aware_fit_on_analytic_affine_field(self):
        rng=np.random.default_rng(24); x=rng.uniform(-1,1,(80,3))
        p,w=voxel_quadrature(x,.12,3)
        # Trace-free affine field; analytic average equals its value at the centre.
        value=np.column_stack([x[:,0]+2*x[:,1]+1,-x[:,1]+x[:,2],x[:,0]])
        obs=Observations.make([0,10],p,w,np.stack([value,value]),common_validation_mask(80,4),
                              np.arange(80),0,2)
        boundary=x[:8]; inlet=x[8:16]
        def exact(z):return np.column_stack([z[:,0]+2*z[:,1]+1,-z[:,1]+z[:,2],z[:,0]])
        for vector in [False,True]:
            models,audit=fit_rbf_pair(obs,geometry_centres(x,12),boundary,np.stack([exact(boundary)]*2),
                                     inlet,np.stack([exact(inlet)]*2),vector,[.8],[0.],max_iter=1000,tol=1e-11)
            q=rng.uniform(-.8,.8,(15,3))
            np.testing.assert_allclose(models[0].predict(q),exact(q),atol=2e-7)
            self.assertFalse(audit['selection_uses_hidden_targets'])
            self.assertTrue(all(t['converged'] for t in audit['trials']))


@unittest.skipUnless(HAS_TORCH,'PyTorch tested in the Brosnan scientific environment')
class DifferentiableOperatorTests(unittest.TestCase):
    def test_quadratic_average_and_parameter_gradient(self):
        import torch
        class Quadratic(torch.nn.Module):
            def __init__(self):
                super().__init__();self.a=torch.nn.Parameter(torch.tensor(2.,dtype=torch.float64))
            def forward(self,x,t):
                return torch.cat([self.a*x*x,torch.zeros_like(t)],dim=1)
        centres=np.array([[.1,.2,.3],[-.2,.5,.6]])
        p,w=voxel_quadrature(centres,.4,3)
        net=Quadratic()
        predicted=torch_observe(net,torch.tensor(p),torch.tensor(w),0)
        analytic=centres**2+.4**2/12
        np.testing.assert_allclose(predicted.detach().numpy(),2*analytic,atol=1e-14)
        predicted.sum().backward()
        self.assertAlmostEqual(float(net.a.grad),float(analytic.sum()),places=13)

    @unittest.skipUnless(importlib.util.find_spec('idealaorta_pinn') is not None,
                         'set PYTHONPATH to the historical package for the training smoke test')
    def test_actual_data_and_continuity_training_paths(self):
        import torch
        from neural import fit_neural
        torch.set_num_threads(2)
        x=np.random.default_rng(71).normal(size=(32,3))*.01
        p,w=voxel_quadrature(x,.001,2)
        values=np.stack([np.ones_like(x)*.1,np.ones_like(x)*.11])
        obs=Observations.make([1775,1780],p,w,values,common_validation_mask(32,4),np.arange(32),.01,8)
        original=obs.values.copy()
        problem=SimpleNamespace(X=x,outlet_xyz=x[:4])
        for arm in ['data','cont']:
            model,audit=fit_neural(problem,obs,x[:8],x[8:12],values[:,8:12],arm,1,0,
                                   device='cpu',obs_batch=4,col_batch=4,wall_batch=8,inlet_batch=4)
            self.assertEqual(model.predict(x[:3],1780).shape,(3,3))
            self.assertEqual(audit['steps'],1)
            self.assertTrue(np.isfinite(audit['history'][0]['loss']))
        np.testing.assert_array_equal(obs.values,original)

    @unittest.skipUnless(importlib.util.find_spec('idealaorta_pinn') is not None,
                         'historical architecture required')
    def test_measurements_only_neural_paths_do_not_require_or_invent_boundaries(self):
        import torch
        from neural import fit_neural
        torch.set_num_threads(2)
        x=np.random.default_rng(242).normal(size=(32,3))*.01
        support,weights=voxel_quadrature(x,.001,2)
        values=np.stack([np.ones_like(x)*.1,np.ones_like(x)*.11])
        obs=Observations.make([1775,1780],support,weights,values,common_validation_mask(32,4),np.arange(32),.01,8)
        problem=SimpleNamespace(X=x)  # Deliberately has no outlet geometry/pressure information.
        for arm in ('data','cont'):
            model,receipt=fit_neural(problem,obs,None,None,None,arm,1,0,device='cpu',
                                     obs_batch=4,col_batch=4,measurements_only=True)
            self.assertTrue(receipt['boundary_contract'].startswith('measurements only'))
            self.assertEqual(receipt['history'][0]['wall'],0.)
            self.assertEqual(receipt['history'][0]['inlet'],0.)
            self.assertTrue(np.isfinite(model.predict(x[:2],1780)).all())


if __name__=='__main__':unittest.main()
