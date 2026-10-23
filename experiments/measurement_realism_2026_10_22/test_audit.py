"""Scientific audit regressions: stale receipts must not certify new results."""
from pathlib import Path
import tempfile
import unittest
import json
import copy
import numpy as np
from assess_pilot import assess_single_draw
from observations import Observations,common_validation_mask
from seal_protocol import validate_capacity,validate_external_contract


class AuditTests(unittest.TestCase):
    def test_actual_adapter_assessment_can_pass_seal_and_reject_changed_direct_solver(self):
        from assess_public_adapter import assess,EXECUTION_SOURCES
        from seal_protocol import sha
        root=Path(__file__).parent
        with tempfile.TemporaryDirectory(prefix='idealaorta_adapter_handover_') as temp:
            folder=Path(temp);unit=folder/'draw00'
            for kind in ('observations','results','models'):(unit/kind).mkdir(parents=True)
            historical=folder/'historical';(historical/'idealaorta_pinn').mkdir(parents=True)
            (historical/'idealaorta_pinn/fields.py').write_text('# manufactured provenance fixture\n')
            historical_hashes={'idealaorta_pinn/fields.py':sha(historical/'idealaorta_pinn/fields.py')}
            source_hashes={p.name:sha(p) for p in root.glob('*.py')}
            axis={'status':'development_axis_calibration_pass','evaluation_files_accessed':False,
                  'audit_source_sha256':source_hashes['audit_external_axes.py'],'source_sha256':'fixture'}
            axis_path=folder/'axis.json';axis_path.write_text(json.dumps(axis))
            protocol={'role':'adapter development','sha256':'fixture_execution',
                      'config':{'file':'aorta01_HR.h5','historical_root':str(historical),
                                'source_hashes':source_hashes,'historical_source_hashes':historical_hashes},
                      'contract':{'steps':100}}
            (folder/'execution_protocol.json').write_text(json.dumps(protocol))
            (unit/'operator_receipt.json').write_text(json.dumps({'source':{'source_sha256':'fixture'},
                'quadrature_relative_difference_per_frame':[0.,0.]}))
            points=np.arange(30,dtype=float).reshape(10,3)
            for sigma in (0.,.02):
                obs=Observations.make([100,110],points[:,None],np.ones(1),
                    np.ones((2,10,3)),common_validation_mask(10,1),np.arange(10),sigma,1)
                for kind in ('point_core','volume_core'):
                    condition=f'{kind}_sigma{sigma:g}';path=unit/'observations'/(condition+'.npz')
                    np.savez(path,support_m=obs.support_m,weights=obs.weights,values=obs.values,
                        clean=obs.clean,validation=obs.validation,anchor_ids=obs.anchor_ids)
                    path.with_suffix('.json').write_text(json.dumps(obs.metadata))
                    if sigma:continue
                    for method in ('data','cont','scalar_rbf','df_rbf'):
                        suffix='.pt' if method in ('data','cont') else '.npz'
                        model=unit/'models'/f'{condition}__{method}{suffix}'
                        model.write_bytes(b'manufactured provenance fixture; never loaded as a model')
                        record={'protocol_sha256':'fixture_execution','role':'adapter development',
                            'status':'complete','clean_targets_unchanged':True,'observations':obs.metadata,
                            'fit':{'boundary_contract':'measurements only'},
                            'scores':{'pressure_WSS_evaluated':False,'boundary_targets_available':False},
                            'model_sha256':sha(model)}
                        (unit/'results'/f'{condition}__{method}.json').write_text(json.dumps(record))
            (folder/'terminal.json').write_text(json.dumps({'results':8,'failures':0,'protocol_sha256':'fixture_execution'}))
            assessment=assess(folder,axis_path)
            self.assertEqual(assessment['status'],'external_adapter_development_checks_pass',assessment['issues'])
            design={'track_A_public_velocity':{'execution_contract':{
                'steps':20000,'max_score_nodes':0,'draws':[0,1,2],
                'methods':['data','cont','scalar_rbf','df_rbf'],
                'conditions':[f'{operator}_sigma{sigma}' for operator in ('point_all','point_core','volume_core')
                              for sigma in ('0','0.02')],
                'measurements_only':True,'no_all_observation_refit':True}}}
            validate_external_contract(design,assessment,root)
            self.assertEqual(assessment['execution_source_hashes']['direct_ridge.py'],sha(root/'direct_ridge.py'))
            changed=copy.deepcopy(assessment);changed['execution_source_hashes']['direct_ridge.py']='changed'
            with self.assertRaisesRegex(ValueError,'differs from the proposed execution source'):
                validate_external_contract(design,changed,root)

    def test_changed_saved_observations_are_detected_without_result_records(self):
        rng=np.random.default_rng(234);points=rng.normal(size=(10,3))
        obs=Observations.make([0,10],points[:,None],np.ones(1),rng.normal(size=(2,10,3)),
                              common_validation_mask(10,1),np.arange(10),0,1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'case01_t1780/observations';folder.mkdir(parents=True)
            path=folder/'point_all_sigma0.npz'
            changed=obs.values.copy();changed[0,0,0]+=.1
            np.savez(path,support_m=obs.support_m,weights=obs.weights,values=changed,
                     clean=obs.clean,validation=obs.validation,anchor_ids=obs.anchor_ids)
            path.with_suffix('.json').write_text(json.dumps(obs.metadata))
            audit=assess_single_draw(root,2000)
            self.assertEqual(audit['status'],'incomplete_or_failed')
            self.assertTrue(any('saved observation hash/counts disagree' in issue for issue in audit['issues']))

    def test_partial_comparator_receipt_cannot_seal_protocol(self):
        receipt={'status':'boundary_gates_pass','results':[
            {'condition':'point_all_sigma0.02','vector':True,'status':'boundary_gate_pass'}]}
        with self.assertRaisesRegex(ValueError,'BOTH noisy point and volume'):
            validate_capacity(receipt,Path(__file__).parent)

    def test_unresolved_adapter_or_smoke_budget_cannot_seal_external_protocol(self):
        with self.assertRaisesRegex(ValueError,'remain incomplete'):
            validate_external_contract({'track_A_public_velocity':{}},{'status':'axis_mapping_unresolved'},Path(__file__).parent)
        with self.assertRaisesRegex(ValueError,'not a short adapter run'):
            validate_external_contract({'track_A_public_velocity':{'execution_contract':{'steps':100,'max_score_nodes':5000}}},
                {'status':'external_adapter_development_checks_pass'},Path(__file__).parent)


if __name__=='__main__':unittest.main()
