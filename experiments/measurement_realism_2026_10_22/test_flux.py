import importlib.util
import unittest
import numpy as np
from flux import PlanarCap,cap_balance


@unittest.skipUnless(importlib.util.find_spec('scipy') is not None,'SciPy cap quadrature verified on Brosnan')
class CapTests(unittest.TestCase):
    def test_area_and_linear_flux_on_rotated_rectangle(self):
        x,y=np.meshgrid(np.linspace(-.02,.02,7),np.linspace(-.01,.01,5))
        p=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
        rotation,_=np.linalg.qr(np.random.default_rng(52).normal(size=(3,3)))
        centre=np.array([.03,.04,.05]);p=p@rotation.T+centre
        expected_normal=rotation[:,2]
        cap=PlanarCap.from_points(p,centre-.01*expected_normal)
        self.assertAlmostEqual(cap.weights_m2.sum(),.04*.02,places=14)
        np.testing.assert_allclose(cap.outward_normal,expected_normal,atol=1e-13)
        speed=.3+2*(p-centre)@rotation[:,0]
        self.assertAlmostEqual(cap.integrate(speed[:,None]*expected_normal),.3*.04*.02,places=14)

    def test_cap_balance_and_nonplanar_rejection(self):
        y,z=np.meshgrid(np.linspace(-.01,.01,5),np.linspace(-.01,.01,5))
        points=np.column_stack([np.zeros(y.size),y.ravel(),z.ravel()])
        inlet=PlanarCap.from_points(points,[.01,0,0])
        outlet=PlanarCap.from_points(points+[.1,0,0],[.09,0,0])
        score=cap_balance(lambda x:np.tile([.2,0,0],(len(x),1)),inlet,outlet)
        self.assertAlmostEqual(score['relative_cap_imbalance'],0,places=14)
        warped=points.copy();warped[0,0]=.001
        with self.assertRaises(ValueError):PlanarCap.from_points(warped,[.01,0,0])


if __name__=='__main__':unittest.main()
