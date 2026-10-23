import unittest
import numpy as np
from operators import voxel_quadrature
from voxel_adapter import VoxelDomain,LinearGridVelocity


class VoxelAdapterTests(unittest.TestCase):
    def test_full_box_certificate_finds_internal_holes_not_only_box_corners(self):
        mask=np.ones((15,15,15),bool);mask[7,7,7]=False
        domain=VoxelDomain(mask,np.array([.5,1.,2.]),np.array([10.,20.,30.]))
        centres=domain.origin_m+np.array([[7,7,7],[3,3,3],[.5,.5,.5]])*domain.spacing_m
        result=domain.certify_boxes(centres,np.array([2.,4.,8.]),margin_m=0)
        np.testing.assert_array_equal(result,[False,True,False])
        untouched=VoxelDomain(np.ones_like(mask),domain.spacing_m,domain.origin_m)
        self.assertTrue(untouched.certify_boxes(centres[:1],np.array([2.,4.,8.]),0)[0])

    def test_certificate_matches_exhaustive_cell_check_and_sampling_uses_only_geometry(self):
        rng=np.random.default_rng(236);mask=rng.random((12,13,14))>.01
        domain=VoxelDomain(mask,[.3,.5,.7],[1,2,3]);width=np.array([.3,.6,.5])
        centres=domain.origin_m+rng.uniform(0,[3.3,6,9.1],(200,3));margin=.01
        actual=domain.certify_boxes(centres,width,margin)
        expected=[]
        for x in centres:
            low=np.floor((x-width/2-margin-domain.origin_m)/domain.spacing_m).astype(int)
            high=np.floor((x+width/2+margin-domain.origin_m)/domain.spacing_m).astype(int)
            ok=np.all(low>=0) and np.all(high<np.array(mask.shape)-1)
            expected.append(ok and mask[low[0]:high[0]+2,low[1]:high[1]+2,low[2]:high[2]+2].all())
        np.testing.assert_array_equal(actual,expected)
        a=domain.cartesian_centres(.7,2);b=domain.cartesian_centres(.7,2)
        np.testing.assert_array_equal(a,b)
        self.assertTrue(np.all(a>=domain.origin_m))
        isolated=np.zeros((4,4,4),bool);isolated[1,1,1]=True
        self.assertEqual(len(VoxelDomain(isolated,np.ones(3),np.zeros(3)).cartesian_centres(.1,0)),0)

    def test_axes_units_and_trilinear_volume_integral_of_manufactured_field(self):
        spacing=np.array([.002,.003,.004]);origin=np.array([.1,-.2,.3])
        xyz=np.stack(np.meshgrid(*(origin[c]+np.arange(8)*spacing[c] for c in range(3)),indexing='ij'),axis=-1)
        def truth(x):return np.stack([x[...,0]*x[...,1],x[...,1]*x[...,2],x[...,0]*x[...,2]],axis=-1)
        transform=np.array([[0,0,-1],[1,0,0],[0,-1,0]])
        stored=truth(xyz)@transform
        field=LinearGridVelocity(np.stack([stored,2*stored]),spacing,origin,transform)
        centres=origin+np.array([[3.4,3.2,3.1],[4.1,2.8,4.3]])*spacing
        p,w=voxel_quadrature(centres,.002,3)
        for frame in (0,1):
            predicted=field.predict(p.reshape(-1,3),frame).reshape(p.shape)
            average=np.einsum('nqc,q->nc',predicted,w)
            np.testing.assert_allclose(average,(frame+1)*truth(centres),atol=1e-14)
        corners=xyz[[0,7],[0,7],[0,7]]
        np.testing.assert_allclose(field.predict(corners,0),truth(corners),atol=1e-14)
        with self.assertRaises(ValueError):field.predict([[100,0,0]],0)
        with self.assertRaises(ValueError):LinearGridVelocity(np.stack([stored]),spacing,origin,np.ones((3,3)))


if __name__=='__main__':unittest.main()
