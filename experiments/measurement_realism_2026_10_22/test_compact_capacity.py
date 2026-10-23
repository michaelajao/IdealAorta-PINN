import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from compact_capacity import assess_basis_plateau
from seal_protocol import validate_capacity,sha


class BasisCapacityTests(unittest.TestCase):
    def levels(self,old=.0101,new=.01):
        return [{'status':'boundary_gate_pass','basis_centres':n,'separation_m':separation,
                 'fit':{'best_val_observed_rel_l2':error}}
                for n,separation,error in [(10,.00125,old),(12,.001,new)]]

    def test_small_error_change_and_prediction_change_are_both_required(self):
        old,new=self.levels();observed=np.ones((2,8,3));previous=observed*1.001;last=observed.copy()
        result=assess_basis_plateau(old,new,previous,last,observed)
        self.assertEqual(result['status'],'capacity_settled')
        # Equal error magnitudes can conceal very different fitted fields.
        result=assess_basis_plateau(old,new,-last,last,observed)
        self.assertEqual(result['status'],'capacity_unsettled')

    def test_large_validation_degradation_is_not_a_plateau(self):
        old,new=self.levels(old=.01,new=.1);y=np.ones((2,8,3))
        self.assertEqual(assess_basis_plateau(old,new,y,y,y)['status'],'capacity_unsettled')
        old,new=self.levels();new['basis_centres']=old['basis_centres']
        self.assertEqual(assess_basis_plateau(old,new,y,y,y)['status'],'capacity_unsettled')
        new['status']='fit_failed'
        self.assertEqual(assess_basis_plateau(old,new,None,None,y)['status'],'capacity_unsettled')

    def test_boundary_only_evidence_cannot_certify_basis_capacity(self):
        receipt={'status':'boundary_gates_pass','results':[
            {'condition':condition,'vector':vector} for condition in ('point_all_sigma0.02','volume_core_sigma0.02')
            for vector in (False,True)]}
        with self.assertRaisesRegex(ValueError,'boundary-only'):
            validate_capacity(receipt,Path(__file__).parent)

    def test_seal_verifies_capacity_artifacts_and_recomputes_plateau(self):
        root=Path(__file__).parent
        source_names=['compact_rbf.py','compact_check.py','compact_capacity.py','observations.py',
                      'operators.py','sparse_backend.py','direct_ridge.py']
        observed=np.ones((2,8,3))
        with tempfile.TemporaryDirectory(prefix='idealaorta_capacity_') as temp:
            folder=Path(temp).resolve();levels=self.levels()
            for index,level in enumerate(levels):
                model=folder/f'model{index}.npz';holdout=folder/f'holdout{index}.npz'
                np.savez(model,centres=np.zeros((level['basis_centres'],3)))
                np.savez(holdout,predicted=observed*(1.001 if index==0 else 1),observed=observed,anchor_ids=np.arange(8))
                level.update(model_file=model.name,holdout_file=holdout.name,model_sha256=sha(model),holdout_sha256=sha(holdout))
                level['fit'].update(boundary_tolerance=.02,selected_candidate={'radius_m':.015,'ridge':1e-9},
                                    selection_uses_hidden_targets=False)
            capacity=assess_basis_plateau(*levels,observed*1.001,observed,observed)
            rows=[]
            for condition in ('point_all_sigma0.02','volume_core_sigma0.02'):
                for vector in (False,True):
                    rows.append({'condition':condition,'vector':vector,'status':'boundary_gate_pass',
                        'fit':levels[-1]['fit'],'model_sha256':levels[-1]['model_sha256'],
                        'observation_file_sha256':'fixture_observations','levels':copy.deepcopy(levels),'capacity':capacity})
            receipt={'status':'capacity_settled','results':rows,'source_hashes':{name:sha(root/name) for name in source_names}}
            validate_capacity(receipt,root,folder)
            changed=copy.deepcopy(receipt);changed['results'][0]['capacity']['relative_holdout_prediction_change']=0
            with self.assertRaisesRegex(ValueError,'statistics differ'):
                validate_capacity(changed,root,folder)
            (folder/'holdout0.npz').write_bytes(b'altered')
            with self.assertRaisesRegex(ValueError,'differs from its receipt'):
                validate_capacity(receipt,root,folder)


if __name__=='__main__':unittest.main()
