import unittest
from unittest.mock import patch
import numpy as np
from public_adapter import authorize_public_file,load_public_window,PublicWindow
from voxel_adapter import VoxelDomain,LinearGridVelocity


class PublicAdapterTests(unittest.TestCase):
    def test_unsealed_evaluation_is_rejected_before_file_or_HDF5_access(self):
        for name in ('aorta02_HR.h5','aorta03trans_HR.h5'):
            with patch('pathlib.Path.stat',side_effect=AssertionError('evaluation file accessed')):
                with self.assertRaisesRegex(ValueError,'sealed independent protocol'):
                    load_public_window(name,'unused_axis_receipt.json')
        with self.assertRaisesRegex(ValueError,'outside the declared'):
            authorize_public_file('aorta03_HR.h5')

    def test_public_operators_pair_locations_noise_and_shared_holdout(self):
        spacing=np.full(3,.001);origin=np.zeros(3)
        x=np.stack(np.meshgrid(*(np.arange(12)*.001 for _ in range(3)),indexing='ij'),axis=-1)
        values=np.stack([x+np.array([.2,.1,.05]),2*x+np.array([.2,.1,.05])])
        window=PublicWindow(LinearGridVelocity(values,spacing,origin,np.eye(3)),
                            VoxelDomain(np.ones((12,12,12),bool),spacing,origin),(100,110),{'role':'manufactured adapter test'})
        conditions,receipt=window.prepare(0)
        self.assertEqual(len(conditions),6)
        self.assertEqual(receipt['quadrature_selected_order'],3)
        for sigma in ('0','0.02'):
            p=conditions['point_core_sigma'+sigma];v=conditions['volume_core_sigma'+sigma]
            np.testing.assert_array_equal(p.anchor_ids,v.anchor_ids)
            np.testing.assert_array_equal(p.validation,v.validation)
            np.testing.assert_allclose(p.values-p.clean,v.values-v.clean,atol=1e-15)
            # The average of an affine field is its value at the box centre.
            np.testing.assert_allclose(p.clean,v.clean,atol=1e-14)

    def test_public_scores_match_exact_affine_truth_without_boundary_targets(self):
        from run_public_pilot import scores
        spacing=np.full(3,.001);origin=np.zeros(3)
        xyz=np.stack(np.meshgrid(*(np.arange(12)*.001 for _ in range(3)),indexing='ij'),axis=-1)
        def exact(x,f):return (f+1)*x+np.array([.2,.1,.05])
        window=PublicWindow(LinearGridVelocity(np.stack([exact(xyz,0),exact(xyz,1)]),spacing,origin,np.eye(3)),
                            VoxelDomain(np.ones((12,12,12),bool),spacing,origin),(100,110),{})
        conditions,_=window.prepare(0);obs=conditions['volume_core_sigma0']
        coords=xyz.reshape(-1,3)
        predictors={t:lambda x,f=f:exact(x,f) for f,t in enumerate(window.times_ms)}
        result=scores(window,obs,predictors,coords,np.full(len(coords),.004))
        self.assertLess(result['pooled_relative_vector_L2'],1e-14)
        self.assertLess(result['acceleration_error_vector_RMSE_m_s2'],1e-12)
        self.assertLess(result['common_grid_divergence']['error_rms_per_s'],1e-10)
        self.assertFalse(result['pressure_WSS_evaluated'])
        self.assertFalse(result['boundary_targets_available'])
