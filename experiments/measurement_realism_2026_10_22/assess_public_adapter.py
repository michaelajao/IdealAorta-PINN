"""Verify public adapter integration without certifying scientific accuracy.

This gate covers aorta01, both point/average operators and all four methods.
It cannot certify the native pilot, comparator capacity or held-out evaluation.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from public_adapter import file_hash
from observations import array_hash


EXECUTION_SOURCES=['public_adapter.py','run_public_pilot.py','voxel_adapter.py','audit_external_axes.py',
                   'observations.py','operators.py','neural.py','compact_rbf.py','sparse_backend.py',
                   'direct_ridge.py','run_pilot.py','historical.py']


def assess(root,axis_audit):
    root=Path(root);source=Path(__file__).parent;issues=[];records=[]
    protocol_path=root/'execution_protocol.json'
    if not protocol_path.exists():raise ValueError('public adapter execution protocol is missing')
    protocol=json.loads(protocol_path.read_text());config=protocol['config']
    if protocol.get('role')!='adapter development' or Path(config['file']).name!='aorta01_HR.h5':
        raise ValueError('adapter gate accepts only the declared development geometry')
    hashes={name:file_hash(source/name) for name in EXECUTION_SOURCES}
    if any(config['source_hashes'].get(name)!=value for name,value in hashes.items()):
        issues.append('execution implementation differs from the proposed adapter')
    historical_root=Path(config['historical_root'])
    historical_hashes={p.relative_to(historical_root).as_posix():file_hash(p)
                       for p in (historical_root/'idealaorta_pinn').rglob('*.py')}
    if not historical_hashes or historical_hashes!=config.get('historical_source_hashes'):
        issues.append('architecture/historical source provenance differs')
    axis_path=Path(axis_audit);axis=json.loads(axis_path.read_text())
    if axis.get('status')!='development_axis_calibration_pass' or axis.get('evaluation_files_accessed') is not False:
        issues.append('development component-axis calibration is unresolved')
    if axis.get('audit_source_sha256')!=hashes['audit_external_axes.py']:
        issues.append('axis-calibration implementation differs')
    unit=root/'draw00';operator=json.loads((unit/'operator_receipt.json').read_text())
    if operator['source'].get('source_sha256')!=axis.get('source_sha256'):
        issues.append('axis calibration and actual observation source differ')
    if max(operator['quadrature_relative_difference_per_frame'])>.01:
        issues.append('volume quadrature convergence failed')
    saved={}
    for sigma in ('0','0.02'):
        for kind in ('point_core','volume_core'):
            condition=f'{kind}_sigma{sigma}';path=unit/'observations'/(condition+'.npz')
            with np.load(path) as data:
                z={key:data[key] for key in data.files}
            metadata=json.loads(path.with_suffix('.json').read_text())
            actual=array_hash(*[z[key] for key in ('support_m','weights','values','validation','anchor_ids')])
            if actual!=metadata['observation_sha256']:issues.append(condition+': saved observation arrays differ')
            saved[condition]=(z,metadata)
        point=saved['point_core_sigma'+sigma][0];volume=saved['volume_core_sigma'+sigma][0]
        if not np.array_equal(point['anchor_ids'],volume['anchor_ids']) or not np.array_equal(point['validation'],volume['validation']):
            issues.append('point/volume centres or observation hold-out differ')
        if not np.allclose(point['values']-point['clean'],volume['values']-volume['clean'],atol=1e-14,rtol=1e-12):
            issues.append('point/volume noise realization differs')
    for condition in ('point_core_sigma0','volume_core_sigma0'):
        for method in ('data','cont','scalar_rbf','df_rbf'):
            path=unit/'results'/f'{condition}__{method}.json'
            if not path.exists():issues.append(condition+'/'+method+': integration result missing');continue
            record=json.loads(path.read_text());records.append({'path':str(path),'sha256':file_hash(path)})
            if record.get('protocol_sha256')!=protocol['sha256'] or record.get('role')!='adapter development':
                issues.append(condition+'/'+method+': result belongs to another execution protocol/role')
            if record.get('status')!='complete' or record.get('clean_targets_unchanged') is not True:
                issues.append(condition+'/'+method+': incomplete or changed clean targets')
            if record['observations']['observation_sha256']!=saved[condition][1]['observation_sha256']:
                issues.append(condition+'/'+method+': different fit information')
            if not record['fit'].get('boundary_contract','').startswith('measurements only'):
                issues.append(condition+'/'+method+': unsupported boundary targets')
            if record['scores'].get('pressure_WSS_evaluated') is not False or record['scores'].get('boundary_targets_available') is not False:
                issues.append(condition+'/'+method+': unsupported endpoint/boundary claim')
            suffix='.pt' if method in ('data','cont') else '.npz'
            model=unit/'models'/f'{condition}__{method}{suffix}'
            if not model.exists() or file_hash(model)!=record.get('model_sha256'):
                issues.append(condition+'/'+method+': saved model provenance differs')
    terminal=root/'terminal.json'
    if not terminal.exists():issues.append('integration worker has no terminal record')
    else:
        finished=json.loads(terminal.read_text())
        if finished.get('results')!=8 or finished.get('failures')!=0 or finished.get('protocol_sha256')!=protocol['sha256']:
            issues.append('integration worker coverage/failure record is incomplete')
    return {'status':'external_adapter_development_checks_pass' if not issues else 'adapter_integration_incomplete_or_failed',
            'issues':issues,'scope':'development input/axis/operator/API integration only; not accuracy or held-out evaluation',
            'expected_integration_records':8,'completed_records':len(records),'records':records,
            'execution_source_hashes':hashes,'historical_source_hashes':historical_hashes,
            'axis_audit_sha256':file_hash(axis_path),'axis_audit':axis,'operator_receipt':operator,
            'execution_protocol_sha256':file_hash(protocol_path),'smoke_contract':protocol['contract'],
            'does_not_prove':['native comparator adequacy','completed native pressure/WSS pilot','independent geometry performance',
                              'native CFD wall identity','anatomical coordinate origin/orientation']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('--axis-audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=assess(args.root,args.axis_audit)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({'status':report['status'],'issues':report['issues'],'completed_records':report['completed_records']},indent=2))
    raise SystemExit(0 if report['status']=='external_adapter_development_checks_pass' else 1)
