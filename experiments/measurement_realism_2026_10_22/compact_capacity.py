"""Observed-data basis-capacity checks for both compact comparator families.

All known native wall/inlet rows are retained at every level. Only basis
centre separation changes. Both finest levels must pass numerical/boundary
gates, and both validation error and held-out predictions must settle within
the declared tolerance. This establishes observed-data capacity, not hidden
pressure/WSS accuracy. Native Cases 1/9 and both phases remain development.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from compact_rbf import fit_compact_pair
from direct_ridge import add_direct_arguments,direct_settings
from observations import Observations
from run_pilot import write_json,file_hash


def assess_basis_plateau(previous,last,previous_prediction,last_prediction,observed,tolerance=.02):
    if not np.isfinite(tolerance) or not 0<tolerance<=.02:
        raise ValueError('basis plateau tolerance must be positive and at most 2%')
    if previous.get('status')!='boundary_gate_pass' or last.get('status')!='boundary_gate_pass':
        return {'status':'capacity_unsettled','reason':'both finest levels must pass numerical and boundary gates'}
    if not 0<previous['basis_centres']<last['basis_centres'] or not previous['separation_m']>last['separation_m']>0:
        return {'status':'capacity_unsettled','reason':'two distinct increasing basis capacities are required'}
    a=np.asarray(previous_prediction);b=np.asarray(last_prediction);y=np.asarray(observed)
    if a.shape!=b.shape or a.shape!=y.shape or not all(np.isfinite(x).all() for x in (a,b,y)):
        raise ValueError('finite paired held-out predictions and observations required')
    old=float(previous['fit']['best_val_observed_rel_l2']);new=float(last['fit']['best_val_observed_rel_l2'])
    if not np.isfinite(old+new) or min(old,new)<0:raise ValueError('finite nonnegative validation errors required')
    validation_change=abs(old-new)/max(new,1e-15)
    prediction_change=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),np.linalg.norm(y),1e-15))
    passed=validation_change<=tolerance and prediction_change<=tolerance
    return {'status':'capacity_settled' if passed else 'capacity_unsettled',
            'relative_validation_change':validation_change,'relative_holdout_prediction_change':prediction_change,
            'tolerance':tolerance,'basis_counts':[previous['basis_centres'],last['basis_centres']],
            'basis_separations_m':[previous['separation_m'],last['separation_m']],
            'scope':'observed hold-out and known boundaries only; no hidden-target capacity claim'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('unit',type=Path);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--basis-separations',type=float,nargs='+',default=[.00125,.001])
    p.add_argument('--support-radii',type=float,nargs='+',default=[.015,.02])
    p.add_argument('--ridges',type=float,nargs='+',default=[1e-7,1e-9])
    p.add_argument('--max-iter',type=int,default=20000);p.add_argument('--max-nnz',type=int,default=600000000)
    p.add_argument('--wall-weight',type=float,default=10.);p.add_argument('--inlet-weight',type=float,default=10.)
    p.add_argument('--wall-tolerance',type=float,default=.02);p.add_argument('--capacity-tolerance',type=float,default=.02)
    p.add_argument('--column-scaling',action='store_true')
    p.add_argument('--solver-method',choices=['lsmr','direct'],default='direct')
    p.add_argument('--solver-device',default='cuda:0');add_direct_arguments(p)
    args=p.parse_args()
    if args.unit.name not in {'case01_t1780','case01_t2400','case09_t1780','case09_t2400'}:
        p.error('capacity checks are restricted to native development Cases 1/9 and both phases')
    separations=args.basis_separations
    if len(separations)<2 or any(not np.isfinite(x) or x<=0 for x in separations) or any(a<=b for a,b in zip(separations,separations[1:])):
        p.error('provide at least two strictly decreasing positive basis separations')
    if not 0<args.capacity_tolerance<=.02 or not 0<args.wall_tolerance<=.02:
        p.error('capacity/boundary tolerances must be positive and at most 2%')
    if args.output_dir.exists():raise ValueError('retain prior capacity records; choose a fresh output directory')
    gpath=args.unit/'common_geometry_indices.npz';g=np.load(gpath)
    args.output_dir.mkdir(parents=True)
    settings=vars(args)|{'unit':str(args.unit),'output_dir':str(args.output_dir),'centre_separation':separations[-1]}
    receipt={'status':'running','arguments':settings,'geometry_sha256':file_hash(gpath),
             'development_unit':{'case':int(args.unit.name[4:6]),'target_ms':int(args.unit.name[8:])},
             'source_hashes':{f.name:file_hash(f) for f in Path(__file__).parent.glob('*.py')},
             'selection_information':'observed velocity and prescribed boundaries only; no hidden velocity/pressure/WSS read',
             'results':[]}
    for condition in ('point_all_sigma0.02','volume_core_sigma0.02'):
        obs_path=args.unit/'observations'/(condition+'.npz');z=np.load(obs_path)
        metadata=json.loads(obs_path.with_suffix('.json').read_text())
        obs=Observations(tuple(z['times_ms'].tolist()),*[z[k] for k in
            ['support_m','weights','clean','values','validation','anchor_ids']],metadata)
        for key in ('support_m','weights','clean','values','validation','anchor_ids'):getattr(obs,key).flags.writeable=False
        for vector in (False,True):
            name=condition+('__df' if vector else '__scalar');levels=[];predictions=[]
            for index,separation in enumerate(separations):
                started=time.monotonic();folder=args.output_dir/f'level{index:02d}';folder.mkdir(exist_ok=True)
                try:
                    models,fit=fit_compact_pair(obs,g['wall_xyz'],g['inlet_xyz'],g['inlet_values'],vector,
                        args.support_radii,args.ridges,max_iter=args.max_iter,max_nnz=args.max_nnz,
                        centre_separation_m=separation,wall_weight=args.wall_weight,inlet_weight=args.inlet_weight,
                        boundary_tolerance=args.wall_tolerance,column_scaling=args.column_scaling,
                        solver_device=args.solver_device,solver_method=args.solver_method,**direct_settings(args))
                    predicted=np.stack([m.observe(obs.support_m[obs.validation],obs.weights) for m in models])
                    model_path=folder/(name+'.npz');prediction_path=folder/(name+'_holdout.npz')
                    np.savez_compressed(model_path,family='compact',centres=models[0].centres,
                        coefficients=np.stack([m.coefficients for m in models]),radius_m=models[0].radius_m,
                        vector=vector,polynomial=models[0].polynomial)
                    np.savez_compressed(prediction_path,predicted=predicted,observed=obs.values[:,obs.validation],
                                        anchor_ids=obs.anchor_ids[obs.validation])
                    level={'status':'boundary_gate_pass','separation_m':separation,'basis_centres':fit['basis_centres'],
                           'fit':fit,'model_file':str(model_path.relative_to(args.output_dir)),
                           'model_sha256':file_hash(model_path),'holdout_file':str(prediction_path.relative_to(args.output_dir)),
                           'holdout_sha256':file_hash(prediction_path)}
                    predictions.append(predicted)
                except Exception as exc:
                    level={'status':'fit_failed','separation_m':separation,'error':str(exc),'trials':getattr(exc,'trials',None)}
                    predictions.append(None)
                level['seconds']=time.monotonic()-started;levels.append(level)
                write_json(folder/(name+'.json'),level)
                print({'event':'compact_capacity_level','condition':condition,'vector':vector,
                       'level':index,'status':level['status'],'separation_m':separation},flush=True)
            capacity=assess_basis_plateau(levels[-2],levels[-1],predictions[-2],predictions[-1],
                                         obs.values[:,obs.validation],args.capacity_tolerance)
            last=levels[-1]
            row={'condition':condition,'vector':vector,'status':last['status'],'fit':last.get('fit',{}),
                 'model_sha256':last.get('model_sha256'),'model_file':last.get('model_file'),
                 'observation_sha256':obs.metadata['observation_sha256'],'observation_file_sha256':file_hash(obs_path),
                 'capacity':capacity,'levels':levels}
            receipt['results'].append(row);write_json(args.output_dir/'receipt.json',receipt)
    receipt['status']='capacity_settled' if all(r['capacity']['status']=='capacity_settled' for r in receipt['results']) else 'capacity_unsettled'
    receipt['scope']='one native development unit; all four units must pass before sealing'
    write_json(args.output_dir/'receipt.json',receipt)
    return 0 if receipt['status']=='capacity_settled' else 1


if __name__=='__main__':raise SystemExit(main())
