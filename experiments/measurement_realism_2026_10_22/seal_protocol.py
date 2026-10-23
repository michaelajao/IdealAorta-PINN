"""Seal the independent design only after the declared development gates pass.

The access declaration is an explicit research record, not technical proof of
all past access. Sealing does not perform, or claim completion of, evaluation.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
from assess_pilot import assess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_capacity(value,source_root,receipt_root=None):
    """A single easy family/condition cannot certify the four-arm comparator."""
    expected={(condition,vector) for condition in ('point_all_sigma0.02','volume_core_sigma0.02') for vector in (False,True)}
    rows=value.get('results',[])
    if {(r.get('condition'),r.get('vector')) for r in rows}!=expected or len(rows)!=4:
        raise ValueError('adequacy requires both comparator families under BOTH noisy point and volume observations')
    if value.get('status')!='capacity_settled':raise ValueError('comparator basis capacity is unresolved; a boundary-only receipt cannot seal it')
    for name in ['compact_rbf.py','compact_check.py','compact_capacity.py','observations.py','operators.py','sparse_backend.py','direct_ridge.py']:
        if value.get('source_hashes',{}).get(name)!=sha(Path(source_root)/name):
            raise ValueError('capacity evidence uses a different comparator or observation implementation')
    for row in rows:
        fit=row.get('fit',{})
        tolerance=fit.get('boundary_tolerance')
        if row.get('status')!='boundary_gate_pass' or tolerance is None or not 0<tolerance<=.02:
            raise ValueError('capacity selection must enforce the known wall and inlet gate')
        if not row.get('model_sha256') or not row.get('observation_file_sha256'):
            raise ValueError('capacity evidence must identify its model and exact saved observations')
        if not fit.get('selected_candidate') or fit.get('selection_uses_hidden_targets') is not False:
            raise ValueError('capacity selection provenance is incomplete')
        levels=row.get('levels',[])
        if len(levels)<2 or row.get('capacity',{}).get('status')!='capacity_settled':
            raise ValueError('two distinct settled basis levels are required for every comparator/operator')
        if receipt_root is None:raise ValueError('saved capacity models and hold-out predictions must be verified')
        import numpy as np
        from compact_capacity import assess_basis_plateau
        root=Path(receipt_root).resolve();heldouts=[]
        for level in levels[-2:]:
            paths={}
            for kind in ('model','holdout'):
                path=(root/level[kind+'_file']).resolve()
                if not path.is_relative_to(root) or sha(path)!=level[kind+'_sha256']:
                    raise ValueError('capacity model/hold-out artifact differs from its receipt')
                paths[kind]=path
            with np.load(paths['model']) as model:
                if len(model['centres'])!=level['basis_centres']:
                    raise ValueError('saved capacity model has a different basis count')
            with np.load(paths['holdout']) as data:
                heldouts.append({k:data[k].copy() for k in ('predicted','observed','anchor_ids')})
        before,after=heldouts
        if not np.array_equal(before['anchor_ids'],after['anchor_ids']) or not np.array_equal(before['observed'],after['observed']):
            raise ValueError('capacity levels must use exactly the same held-out observations')
        actual=assess_basis_plateau(levels[-2],levels[-1],before['predicted'],after['predicted'],after['observed'],
                                   row['capacity'].get('tolerance'))
        if actual['status']!='capacity_settled' or row['fit']!=levels[-1].get('fit') or row['model_sha256']!=levels[-1].get('model_sha256'):
            raise ValueError('saved basis predictions do not establish the claimed selected capacity')
        for key in ('relative_validation_change','relative_holdout_prediction_change'):
            if not np.isclose(actual[key],row['capacity'].get(key,np.nan),rtol=1e-12,atol=1e-14):
                raise ValueError('capacity plateau statistics differ from saved predictions')


def validate_external_contract(design,adapter,source_root):
    """Refuse a protocol that still has an unresolved external implementation."""
    if adapter.get('status')!='external_adapter_development_checks_pass':
        raise ValueError('external adapter integration/axis/provenance checks remain incomplete')
    contract=design['track_A_public_velocity'].get('execution_contract')
    if not contract or contract.get('steps',0)<20000 or contract.get('max_score_nodes')!=0:
        raise ValueError('seal the final external budget and full scoring set, not a short adapter run')
    if contract.get('draws')!=[0,1,2] or set(contract.get('methods',[]))!={'data','cont','scalar_rbf','df_rbf'}:
        raise ValueError('external contract must retain all declared draws and methods')
    expected={f'{operator}_sigma{sigma}' for operator in ('point_all','point_core','volume_core') for sigma in ('0','0.02')}
    if set(contract.get('conditions',[]))!=expected:raise ValueError('external contract must retain every declared operator/noise condition')
    if contract.get('measurements_only') is not True or contract.get('no_all_observation_refit') is not True:
        raise ValueError('external available-information/holdout contract is incomplete')
    for name in ['public_adapter.py','run_public_pilot.py','voxel_adapter.py','audit_external_axes.py',
                 'observations.py','operators.py','neural.py','compact_rbf.py','sparse_backend.py','direct_ridge.py']:
        if adapter.get('execution_source_hashes',{}).get(name)!=sha(Path(source_root)/name):
            raise ValueError('external adapter evidence differs from the proposed execution source')
    if not adapter.get('axis_audit_sha256') or not adapter.get('historical_source_hashes'):
        raise ValueError('external axis/architecture provenance is missing')


def seal(development_root,capacity_receipts,access_declaration,output,external_adapter_receipt=None):
    output=Path(output)
    if output.exists():raise ValueError('never overwrite a sealed protocol')
    audit=assess(development_root,stage='development')
    if audit['status']!='development_pilot_coverage_and_contracts_pass':
        raise ValueError('the full declared development pilot is incomplete; a smoke/feasibility run cannot seal the protocol')
    capacities=[]
    covered=set()
    root=Path(__file__).parent
    for path in map(Path,capacity_receipts):
        value=json.loads(path.read_text(encoding='utf-8'))
        validate_capacity(value,root,path.parent)
        unit=value.get('development_unit')
        if unit is None:raise ValueError(f'capacity receipt must identify its case and phase: {path}')
        covered.add((unit['case'],unit['target_ms']))
        capacities.append({'path':str(path.resolve()),'sha256':sha(path),'receipt':value})
    if covered!={(1,1780),(1,2400),(9,1780),(9,2400)}:
        raise ValueError('adequacy evidence must cover BOTH development cases and BOTH phases')
    access_path=Path(access_declaration)
    access=json.loads(access_path.read_text(encoding='utf-8'))
    if access.get('external_evaluation_velocity_arrays_accessed') is not False or access.get('external_evaluation_pressure_WSS_accessed') is not False:
        raise ValueError('record/resolve prior evaluation-target access before claiming a prospective seal')
    core=['run_pilot.py','observations.py','operators.py','neural.py','comparators.py','compact_rbf.py',
          'historical.py','geometry.py','flux.py','metrics.py','scoring.py','sparse_backend.py','direct_ridge.py','voxel_adapter.py',
          'public_adapter.py','run_public_pilot.py','audit_external_axes.py']
    execution_configs={}
    for draw in (0,1,2):
        manifest=Path(development_root)/f'draw{draw:02d}'/'execution_protocol.json'
        config=json.loads(manifest.read_text(encoding='utf-8'))['config']
        if any(config['source_hashes'].get(name)!=sha(root/name) for name in core):
            raise ValueError('the proposed core implementation differs from the completed development pilot')
        execution_configs[str(draw)]=config
    capacity_keys={'support_radii':'support_radii','ridges':'ridges','max_iter':'rbf_iterations',
                   'max_nnz':'compact_max_nnz','centre_separation':'compact_centre_separation',
                   'wall_weight':'rbf_wall_weight','wall_tolerance':'rbf_boundary_tolerance',
                   'inlet_weight':'rbf_inlet_weight',
                   'column_scaling':'compact_column_scaling','solver_device':'rbf_solver_device',
                   'solver_method':'rbf_solver_method','direct_factor_device':'direct_factor_device',
                   'direct_cpu_factor_backend':'direct_cpu_factor_backend',
                   'direct_block_columns':'direct_block_columns','direct_max_gpu_gib':'direct_max_gpu_gib',
                   'direct_max_host_gib':'direct_max_host_gib'}
    for receipt in capacities:
        settings=receipt['receipt'].get('arguments',{})
        for config in execution_configs.values():
            if config.get('rbf_family')!='compact' or any(settings.get(a)!=config.get(b) for a,b in capacity_keys.items()):
                raise ValueError('comparator settings differ between adequacy checks and completed development runs')
    design_path=root/'independent_protocol.json'
    design=json.loads(design_path.read_text(encoding='utf-8'))
    if external_adapter_receipt is None:raise ValueError('a verified external-development adapter receipt is required before sealing')
    adapter=json.loads(Path(external_adapter_receipt).read_text(encoding='utf-8'))
    validate_external_contract(design,adapter,root)
    design['status']='scientific design sealed; see bundle for development and source receipts'
    bundle={'status':'sealed_design; evaluation_not_run','sealed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scientific_design':design,'scientific_design_source_sha256':sha(design_path),
            'scientific_design_sha256':hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest(),
            'source_hashes':{p.name:sha(p) for p in root.glob('*.py')},
            'development_root':str(Path(development_root).resolve()),'development_audit':audit,
            'development_execution_configs':execution_configs,
            'comparator_adequacy':capacities,'prior_access_declaration':access,
            'external_adapter':adapter,'external_adapter_receipt_sha256':sha(external_adapter_receipt),
            'prior_access_declaration_sha256':sha(access_path),
            'post_seal_changes':'version and report every change; outcome-informed changes require a fresh evaluation set'}
    encoded=json.dumps(bundle,sort_keys=True,separators=(',',':')).encode()
    bundle['bundle_sha256']=hashlib.sha256(encoded).hexdigest()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(bundle,stream,indent=2)
    return bundle


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development-root',type=Path,required=True)
    p.add_argument('--capacity-receipts',type=Path,nargs='+',required=True)
    p.add_argument('--access-declaration',type=Path,required=True)
    p.add_argument('--external-adapter-receipt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(seal(args.development_root,args.capacity_receipts,args.access_declaration,args.output,args.external_adapter_receipt),indent=2))
