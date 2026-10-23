"""Velocity-only public-data pilot; unsealed execution is aorta01 development.

Short/sampled runs are adapter integration evidence. They do not complete the
native pressure/WSS pilot or constitute held-out geometry evaluation.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import traceback
import numpy as np
from public_adapter import authorize_public_file,load_public_window,file_hash
from observations import array_hash
from compact_rbf import fit_compact_pair
from direct_ridge import add_direct_arguments,direct_settings
from run_pilot import write_json


def compute_contract(args):
    keys=['steps','support_radii','ridges','rbf_iterations','rbf_max_nnz','centre_separation',
          'solver_device','solver_method','direct_factor_device','direct_cpu_factor_backend','direct_block_columns',
          'direct_max_gpu_gib','direct_max_host_gib','device','max_score_nodes','methods','conditions','draws']
    return {key:getattr(args,key) for key in keys}|{'measurements_only':True,'no_all_observation_refit':True}


def validate_evaluation_settings(path,seal,contract,historical_hashes=None):
    role=authorize_public_file(path,seal)
    if role=='sealed external evaluation':
        bundle=json.loads(Path(seal).read_text(encoding='utf-8'))
        expected=bundle['scientific_design']['track_A_public_velocity'].get('execution_contract')
        if expected!=contract:raise ValueError('external execution settings are not the exact sealed contract')
        if historical_hashes!=bundle.get('external_adapter',{}).get('historical_source_hashes'):
            raise ValueError('historical architecture source differs from the sealed external implementation')
    return role


def scores(window,obs,predictors,coordinates,mask_distances_m):
    predicted=np.stack([predictors[t](coordinates) for t in window.times_ms])
    truth=np.stack([window.field.predict(coordinates,f) for f in (0,1)])
    error=predicted-truth;lengths=np.linalg.norm(error,axis=2)
    observed=np.stack([np.einsum('nqc,q->nc',predictors[t](obs.support_m.reshape(-1,3)).reshape(obs.support_m.shape),obs.weights)
                       for t in window.times_ms])
    dt=(window.times_ms[1]-window.times_ms[0])/1000
    div={}
    probes=coordinates[np.linspace(0,len(coordinates)-1,min(512,len(coordinates)),dtype=int)]
    for eps in (1e-5,5e-6):
        x=probes;d=np.zeros(len(x))
        for c in range(3):
            shift=np.eye(3)[c]*eps
            d+=(predictors[window.times_ms[-1]](x+shift)[:,c]-predictors[window.times_ms[-1]](x-shift)[:,c])/(2*eps)
        div[str(eps)]=float(np.sqrt(np.mean(d*d)))
    bins=[]
    for low,high in [(0,.00125),(.00125,.0025),(.0025,.005),(.005,np.inf)]:
        selected=(mask_distances_m>=low)&(mask_distances_m<high)
        bins.append({'lower_m':low,'upper_m':None if not np.isfinite(high) else high,'nodes':int(selected.sum()),
                     'pooled_relative_vector_L2':float(np.linalg.norm(error[:,selected])/np.linalg.norm(truth[:,selected]))
                         if selected.any() and np.linalg.norm(truth[:,selected]) else None})
    x=probes;spacing=window.domain.spacing_m
    x=x[window.domain.certify_boxes(x,2*spacing,margin_m=1e-10)]
    model_div=np.zeros(len(x));reference_div=np.zeros(len(x))
    for c in (range(3) if len(x) else ()):
        shift=np.eye(3)[c]*spacing[c]
        model_div+=(predictors[window.times_ms[-1]](x+shift)[:,c]-predictors[window.times_ms[-1]](x-shift)[:,c])/(2*spacing[c])
        reference_div+=(window.field.predict(x+shift,1)[:,c]-window.field.predict(x-shift,1)[:,c])/(2*spacing[c])
    return {'pooled_relative_vector_L2':float(np.linalg.norm(error)/np.linalg.norm(truth)),
            'per_frame_relative_vector_L2':(np.linalg.norm(error,axis=(1,2))/np.linalg.norm(truth,axis=(1,2))).tolist(),
            'vector_RMSE_m_s':float(np.sqrt(np.mean(lengths**2))),
            'vector_error_95th_percentile_m_s':float(np.quantile(lengths,.95)),
            'acceleration_error_vector_RMSE_m_s2':float(np.sqrt(np.mean(np.sum(((error[1]-error[0])/dt)**2,axis=1)))),
            'finite_difference_divergence_rms_per_s':div,
            'common_grid_divergence':{'stencil_spacing_m':spacing.tolist(),'interior_nodes':len(x),
                  'predicted_rms_per_s':float(np.sqrt(np.mean(model_div**2))) if len(x) else None,
                  'reference_rms_per_s':float(np.sqrt(np.mean(reference_div**2))) if len(x) else None,
                  'error_rms_per_s':float(np.sqrt(np.mean((model_div-reference_div)**2))) if len(x) else None},
            'mask_distance_bins':bins,'distance_definition':'distance to nearest background voxel centre; image-grid proxy, not native wall distance',
            'observation_relative_L2':{name:float(np.linalg.norm((observed-obs.values)[:,mask])/np.linalg.norm(obs.values[:,mask]))
                                      for name,mask in [('fit',~obs.validation),('validation',obs.validation)]},
            'scoring_coordinates_sha256':array_hash(coordinates),'score_nodes':len(coordinates),
            'pressure_WSS_evaluated':False,'boundary_targets_available':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--file',type=Path,required=True);p.add_argument('--axis-audit',type=Path,required=True)
    p.add_argument('--historical-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seal',type=Path);p.add_argument('--rows',nargs=2,type=int,default=[9,10])
    p.add_argument('--draws',nargs='+',type=int,default=[0])
    p.add_argument('--steps',type=int,default=20000);p.add_argument('--device',default='cuda')
    p.add_argument('--methods',nargs='+',choices=['data','cont','scalar_rbf','df_rbf'],default=['data','cont','scalar_rbf','df_rbf'])
    p.add_argument('--conditions',nargs='+',default=['point_all_sigma0','point_all_sigma0.02','point_core_sigma0',
                   'point_core_sigma0.02','volume_core_sigma0','volume_core_sigma0.02'])
    p.add_argument('--support-radii',nargs='+',type=float,default=[.015,.02])
    p.add_argument('--ridges',nargs='+',type=float,default=[1e-6,1e-7]);p.add_argument('--rbf-iterations',type=int,default=20000)
    p.add_argument('--rbf-max-nnz',type=int,default=600000000);p.add_argument('--centre-separation',type=float,default=.001)
    p.add_argument('--solver-device',default='cuda:0');p.add_argument('--max-score-nodes',type=int,default=0)
    p.add_argument('--solver-method',choices=['lsmr','direct'],default='lsmr')
    add_direct_arguments(p)
    args=p.parse_args()
    if args.steps<1 or args.max_score_nodes<0 or not set(args.draws)<={0,1,2}:p.error('invalid declared budget/draws')
    allowed={f'{operator}_sigma{sigma}' for operator in ('point_all','point_core','volume_core') for sigma in ('0','0.02')}
    if not set(args.conditions)<=allowed or len(args.draws)!=len(set(args.draws)) or len(args.methods)!=len(set(args.methods)):
        p.error('unknown conditions or repeated draws/methods')
    args.historical_root=args.historical_root.resolve()
    historical_hashes={f.relative_to(args.historical_root).as_posix():file_hash(f)
                       for f in (args.historical_root/'idealaorta_pinn').rglob('*.py')}
    if not historical_hashes:raise ValueError('historical architecture source is unavailable')
    contract=compute_contract(args);role=validate_evaluation_settings(args.file,args.seal,contract,historical_hashes)
    if args.output.exists():raise ValueError('retain prior public pilot outputs; use a fresh output root')
    args.historical_root=args.historical_root.resolve();args.output=args.output.resolve()
    if args.output==args.historical_root or args.historical_root in args.output.parents:
        raise ValueError('outputs must stay outside the historical worktree')
    sys.path.insert(0,str(args.historical_root))
    source_hashes={f.name:file_hash(f) for f in Path(__file__).parent.glob('*.py')}
    config=vars(args)|{'file':str(args.file),'axis_audit':str(args.axis_audit),'historical_root':str(args.historical_root),
                       'output':str(args.output),'seal':str(args.seal) if args.seal else None,'source_hashes':source_hashes,
                       'historical_source_hashes':historical_hashes,
                       'software':{name:importlib.metadata.version(name) for name in ('numpy','scipy','torch','h5py')},
                       'hardware':{'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'cpu_count':os.cpu_count()}}
    encoded=json.dumps(config,sort_keys=True).encode();protocol_sha=hashlib.sha256(encoded).hexdigest()
    write_json(args.output/'execution_protocol.json',{'sha256':protocol_sha,'config':config,'contract':contract,'role':role})
    window=load_public_window(args.file,args.axis_audit,args.rows,args.seal)
    clean_hash=array_hash(window.field.values);coordinates=window.fluid_coordinates
    from scipy.ndimage import distance_transform_edt
    mask_distances=distance_transform_edt(window.domain.mask,sampling=window.domain.spacing_m)[window.domain.mask]
    if args.max_score_nodes:
        ids=np.sort(np.random.default_rng(42).choice(len(coordinates),min(args.max_score_nodes,len(coordinates)),replace=False))
        scoring=coordinates[ids]
        scoring_distances=mask_distances[ids]
    else:scoring=coordinates;scoring_distances=mask_distances
    np.savez_compressed(args.output/'scoring_geometry.npz',coordinates_m=scoring)
    problem=SimpleNamespace(X=coordinates)
    for draw in args.draws:
        unit=args.output/f'draw{draw:02d}';conditions,operator=window.prepare(draw)
        write_json(unit/'operator_receipt.json',operator)
        for name,obs in conditions.items():
            # Keep directory creation explicit before NumPy writes.
            (unit/'observations').mkdir(parents=True,exist_ok=True)
            np.savez_compressed(unit/'observations'/f'{name}.npz',support_m=obs.support_m,weights=obs.weights,clean=obs.clean,
                                values=obs.values,validation=obs.validation,anchor_ids=obs.anchor_ids,times_ms=obs.times_ms)
            write_json(unit/'observations'/f'{name}.json',obs.metadata)
            if name not in args.conditions:continue
            for method in args.methods:
                started=time.monotonic()
                try:
                    print({'event':'public_fit_started','draw':draw,'condition':name,'method':method},flush=True)
                    folder=unit/'models';folder.mkdir(exist_ok=True)
                    if method in ('data','cont'):
                        import torch
                        from neural import fit_neural
                        field,fit=fit_neural(problem,obs,None,None,None,method,args.steps,draw,device=args.device,measurements_only=True)
                        predictors={t:lambda x,t=t:field.predict(x,t) for t in window.times_ms}
                        model_path=folder/f'{name}__{method}.pt'
                        torch.save({'state_dict':field.net.state_dict(),'scales':field.scales,'fit':fit},model_path)
                    else:
                        empty=np.empty((0,3));inlet_values=np.empty((2,0,3))
                        models,fit=fit_compact_pair(obs,empty,empty,inlet_values,method=='df_rbf',args.support_radii,args.ridges,
                            max_iter=args.rbf_iterations,max_nnz=args.rbf_max_nnz,centre_separation_m=args.centre_separation,
                            column_scaling=True,solver_device=args.solver_device,measurements_only=True,
                            solver_method=args.solver_method,**direct_settings(args))
                        predictors={t:models[f].predict for f,t in enumerate(window.times_ms)}
                        model_path=folder/f'{name}__{method}.npz'
                        np.savez_compressed(model_path,family='compact',centres=models[0].centres,
                            coefficients=np.stack([m.coefficients for m in models]),radius_m=models[0].radius_m,
                            vector=models[0].vector,polynomial=models[0].polynomial)
                    result=scores(window,obs,predictors,scoring,scoring_distances)
                    assert array_hash(window.field.values)==clean_hash,'clean velocity targets changed'
                    write_json(unit/'results'/f'{name}__{method}.json',{'status':'complete','role':role,'method':method,
                        'condition':name,'draw':draw,'observations':obs.metadata,'fit':fit,'scores':result,
                        'scientific_scope':'adapter integration only' if args.steps<20000 or args.max_score_nodes else 'velocity reconstruction only',
                        'clean_targets_unchanged':True,'protocol_sha256':protocol_sha,'model_sha256':file_hash(model_path),
                        'seconds':time.monotonic()-started})
                except Exception as exc:
                    write_json(unit/'failures'/f'{name}__{method}.json',{'status':'failed','error':str(exc),
                        'traceback':traceback.format_exc(),'candidate_trials':getattr(exc,'trials',None)})
                    print(traceback.format_exc(),flush=True)
    results=list(args.output.glob('draw*/results/*.json'));failures=list(args.output.glob('draw*/failures/*.json'))
    write_json(args.output/'terminal.json',{'status':'worker_finished','results':len(results),'failures':len(failures),
               'role':role,'pressure_WSS_evaluated':False,'protocol_sha256':protocol_sha})
    return 1 if failures else 0


if __name__=='__main__':raise SystemExit(main())
