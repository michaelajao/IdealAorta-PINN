"""Dated development-only pilot. Never writes historical models/results/masks.

Run with the scientific environment on Brosnan. --score velocity is explicitly
an integration/feasibility run; only --score full evaluates pressure and WSS.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import importlib.metadata
import sys
import time
import traceback
import numpy as np

from observations import Observations, array_hash, common_validation_mask, matched_location_subsets
from operators import voxel_quadrature
from geometry import OrientedPointDomain
from comparators import geometry_centres, fit_rbf_pair
from historical import api_module,constants
from flux import PlanarCap,cap_balance
from metrics import temporal_error_decomposition
from compact_rbf import fit_compact_pair
from direct_ridge import add_direct_arguments,direct_settings
from scoring import make_audited_integrator,absolute_wss_scores


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x),allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def prepare(P, args, output):
    from scipy.interpolate import LinearNDInterpolator
    wall_normals_from_volume=api_module('data').wall_normals_from_volume
    started=time.monotonic()
    ids=P.obs_ids.copy()
    split=common_validation_mask(len(ids),args.split_seed)
    boundary=np.vstack([P.wall_xyz,P.inlet_xyz,P.outlet_xyz])
    normals=np.vstack([P.wall_normals,wall_normals_from_volume(P.inlet_xyz,P.X),
                       wall_normals_from_volume(P.outlet_xyz,P.X)])
    domain=OrientedPointDomain(boundary,normals,args.geometry_margin_m)
    core=domain.certify_boxes(P.X[ids],args.voxel_width_m)
    core_rows=np.flatnonzero(core)
    if not (core & split).any() or not (core & ~split).any():
        raise ValueError('no usable full-box proxy core with a nonempty shared split')
    print({'event':'geometry_prepared','observations':len(ids),'core_boxes':len(core_rows),
           'domain':domain.metadata()},flush=True)
    clean=np.stack([P.snaps[t].uvw[ids] for t in P.times_ms])
    points=np.vstack([P.X,P.wall_xyz])
    values=np.vstack([np.concatenate([P.snaps[t].uvw for t in P.times_ms],axis=1),
                      np.zeros((len(P.wall_xyz),6))])
    print({'event':'building_frozen_linear_CFD_interpolant','points':len(points)},flush=True)
    oracle=LinearNDInterpolator(points,values)
    averages={}
    for order in (3,5,7):
        support,weights=voxel_quadrature(P.X[ids[core]],args.voxel_width_m,order)
        result=np.empty((len(support),6))
        for start in range(0,len(support),64):
            p=support[start:start+64]
            value=oracle(p.reshape(-1,3)).reshape(len(p),p.shape[1],6)
            if not np.isfinite(value).all():
                raise ValueError('certified proxy box leaves the CFD interpolant convex support')
            result[start:start+len(p)]=np.einsum('nqc,q->nc',value,weights)
        averages[order]=result.reshape(len(result),2,3).transpose(1,0,2)
        print({'event':'quadrature_complete','order':order},flush=True)
        if order > 3:
            previous=order-2
            delta=averages[previous]-averages[order]
            errors=np.linalg.norm(delta,axis=(1,2))/np.linalg.norm(averages[order],axis=(1,2))
            if np.all(errors <= args.quadrature_tolerance):
                chosen=previous
                break
    else:
        raise ValueError('volume quadrature did not meet the declared 3/5/7 convergence gate')
    del oracle,points,values
    receipt={'geometry':domain.metadata(),'n_base':len(ids),'n_core':len(core_rows),
             'quadrature_selected_order':chosen,'quadrature_reference_order':order,
             'quadrature_relative_difference_per_frame':errors.tolist(),
             'quadrature_tolerance':args.quadrature_tolerance,
             'operator_oracle':'frozen piecewise-linear interpolation of supplied CFD and zero wall velocities',
             'measurement_claim':'kinematic spatial averages and additive velocity noise; no MRI acquisition claim',
             'seconds':time.monotonic()-started}
    conditions={}
    for sigma in args.sigmas:
        point=Observations.make(P.times_ms,P.X[ids,None],np.ones(1),clean,split,ids,sigma,
                                args.noise_seed,args.noise_correlation,
                                metadata={'operator':'point','base_grid_seed':P.seed})
        stem=f'sigma{sigma:g}'
        conditions['point_all_'+stem]=point
        conditions['point_core_'+stem]=point.subset(core_rows,'core_matched_to_volume')
        support,weights=voxel_quadrature(P.X[ids[core]],args.voxel_width_m,chosen)
        conditions['volume_core_'+stem]=Observations.make(
            P.times_ms,support,weights,averages[chosen],split[core],ids[core],sigma,args.noise_seed,args.noise_correlation,
            metadata={'operator':'physical_full_box_average_in_oriented_point_proxy',**receipt},
            realized_noise=(point.values-point.clean)[:,core])
        if sigma == 0:
            subsets=matched_location_subsets(ids,P.dwall[ids],split,.00125,args.removal_seed)
            for name in ('exclude_near_wall','matched_random_removal'):
                conditions[name+'_'+stem]=point.subset(subsets[name],name)
    output.mkdir(parents=True,exist_ok=True)
    for name,obs in conditions.items():
        np.savez_compressed(output/(name+'.npz'),support_m=obs.support_m,weights=obs.weights,
                            clean=obs.clean,values=obs.values,validation=obs.validation,anchor_ids=obs.anchor_ids,
                            times_ms=obs.times_ms)
        write_json(output/(name+'.json'),obs.metadata)
    write_json(output/'operator_receipt.json',receipt)
    return conditions


def make_caps(P,args):
    from scipy.spatial import cKDTree
    load_block=api_module('data').load_block
    tree=cKDTree(P.X)
    caps=[]
    diameter=.020 if P.case==1 else .026
    for name,d in [('inlet',diameter),('outlet',diameter*.8)]:
        frame=load_block(P.case,name,P.times_ms[0],columns=['x','y','z'])
        points=frame[['x','y','z']].to_numpy(float)
        _,near=tree.query(points.mean(0),k=8)
        cap=PlanarCap.from_points(points,P.X[near].mean(0))
        expected=np.pi*d*d/4
        relative=abs(cap.weights_m2.sum()-expected)/expected
        if relative>args.cap_area_relative_tolerance:
            raise ValueError(f'{name} cap area differs from the known idealized geometry by {relative:.3%}')
        caps.append(cap)
    return caps


def diagnostic_scores(P,obs,predictors,common_hidden,wall_pool,caps,full_predictions=None,inlet_contract=None):
    velocity_scores=api_module('postprocess').velocity_scores
    target=P.target_ms
    evaluation=common_hidden
    # The same geometry-only held-out target set is used for every condition.
    target_values=P.snaps[target].uvw[evaluation]
    pred=full_predictions[target][evaluation] if full_predictions is not None else predictors[target](P.X[evaluation])
    ref=float(np.quantile(np.linalg.norm(target_values,axis=1),.995))
    score={'velocity':velocity_scores(pred,target_values,ref,P.sac[evaluation]),
           'score_mode':'velocity_feasibility_only','pressure_and_WSS_evaluated':False,
           'evaluation_ids_sha256':array_hash(evaluation),'n_evaluation_nodes':len(evaluation)}
    observed=np.stack([np.einsum('nqc,q->nc',predictors[t](obs.support_m.reshape(-1,3)).reshape(obs.support_m.shape),obs.weights)
                       for t in obs.times_ms])
    score['observation_fit']={}
    for label,mask in [('train',~obs.validation),('validation',obs.validation)]:
        score['observation_fit'][label]=float(np.linalg.norm(observed[:,mask]-obs.values[:,mask])/np.linalg.norm(obs.values[:,mask]))
    # Diagnostics do not feed selection; clean acceleration remains a hidden target.
    other=next(t for t in obs.times_ms if t!=target)
    dt=(target-other)/1000
    pred_other=full_predictions[other][evaluation] if full_predictions is not None else predictors[other](P.X[evaluation])
    acceleration=(pred-pred_other)/dt
    true_acc=(P.snaps[target].uvw[evaluation]-P.snaps[other].uvw[evaluation])/dt
    score['acceleration_rms_error_m_s2']=float(np.sqrt(np.mean(np.sum((acceleration-true_acc)**2,axis=1))))
    predictions={target:pred,other:pred_other}
    score['temporal_error_decomposition']=temporal_error_decomposition(
        np.stack([predictions[t] for t in obs.times_ms]),
        np.stack([P.snaps[t].uvw[evaluation] for t in obs.times_ms]),np.asarray(obs.times_ms)/1000)
    sample=P.X[evaluation[:min(256,len(evaluation))]]
    divergences={}
    for eps in (1e-5,5e-6):
        divergence=np.zeros(len(sample))
        for c in range(3):
            offset=np.zeros(3);offset[c]=eps
            divergence+=(predictors[target](sample+offset)[:,c]-predictors[target](sample-offset)[:,c])/(2*eps)
        divergences[str(eps)]=float(np.sqrt(np.mean(divergence**2)))
    score['finite_difference_divergence_rms_per_s']=divergences
    score['wall_velocity_rms_m_s']=float(np.sqrt(np.mean(np.sum(predictors[target](wall_pool)**2,axis=1))))
    if inlet_contract is not None:
        inlet,inlet_values=inlet_contract
        scale=max(float(np.quantile(np.linalg.norm(obs.values[:,~obs.validation],axis=2),.995)),1e-6)
        wall_errors=[float(np.sqrt(np.mean(np.sum(predictors[t](wall_pool)**2,axis=1))))/scale for t in obs.times_ms]
        inlet_errors=[float(np.sqrt(np.mean(np.sum((predictors[t](inlet)-inlet_values[f])**2,axis=1))))/scale
                      for f,t in enumerate(obs.times_ms)]
        score['known_boundary_fit']={'wall_rms_over_observed_U':wall_errors,'inlet_rms_over_observed_U':inlet_errors,
            'wall_nodes_checked':len(wall_pool),'inlet_nodes_checked':len(inlet),
            'scope':'numerical residuals at known boundary points; includes fitting points; not independent validation'}
    score['cap_flux_balance']=cap_balance(predictors[target],*caps)
    return score


def run_problem(args,case,target,root):
    build_problem=api_module('problem').build_problem
    from idealaorta_pinn.config import FULL_DIR
    RHO,MU=constants()
    assert RHO==1050.0 and MU==.0035, 'pilot must use the corrected physical constants'
    times=[1775,1780] if target==1780 else [2390,2400]
    destination=root/f'case{case:02d}_t{target}'
    destination.mkdir(parents=True,exist_ok=True)
    print({'event':'loading_problem','case':case,'target':target},flush=True)
    P=build_problem(case,times,target,2.5,args.grid_seed,save_mask=False)
    clean_hash=array_hash(*[P.snaps[t].uvw for t in times])
    for snapshot in P.snaps.values():
        for f in dataclasses.fields(snapshot):
            a=getattr(snapshot,f.name)
            if isinstance(a,np.ndarray): a.flags.writeable=False
    manifest={p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)}
              for t in times for p in FULL_DIR.glob(f'case{case:02d}_*_t{t}.parquet')}
    write_json(destination/'source_data.json',{'processed_parquet_files':manifest,'clean_velocity_sha256':clean_hash})
    prepared=destination/'observations'
    if (prepared/'operator_receipt.json').exists():
        conditions={}
        for path in sorted(prepared.glob('*.npz')):
            z=np.load(path)
            meta=json.loads(path.with_suffix('.json').read_text())
            arrays=[z[k] for k in ['support_m','weights','clean','values','validation','anchor_ids']]
            for a in arrays:a.flags.writeable=False
            conditions[path.stem]=Observations(tuple(z['times_ms'].tolist()),*arrays,meta)
    else:
        conditions=prepare(P,args,prepared)
    rng=np.random.default_rng(args.boundary_seed)
    wall_count=len(P.wall_xyz) if args.boundary_count==0 else min(args.boundary_count,len(P.wall_xyz))
    wall_ids=np.sort(rng.choice(len(P.wall_xyz),wall_count,replace=False))
    inlet_ids=np.sort(rng.choice(len(P.inlet_xyz),min(512,len(P.inlet_xyz)),replace=False))
    wall=P.wall_xyz[wall_ids];inlet=P.inlet_xyz[inlet_ids]
    inlet_values=np.stack([P.snaps[t].inlet_uvw[inlet_ids] for t in times])
    basis_sequence=geometry_centres(np.vstack([P.X,wall,inlet]),max(args.centres,1024),args.basis_seed)
    centres=basis_sequence[:args.centres]
    remaining=np.setdiff1d(np.arange(len(P.wall_xyz)),wall_ids)
    wall_validation_pool=remaining if len(remaining) else np.arange(len(P.wall_xyz))
    wall_validation_ids=np.sort(rng.choice(wall_validation_pool,min(1024,len(wall_validation_pool)),replace=False))
    caps=make_caps(P,args)
    common_hidden=P.hidden_ids
    if args.score=='velocity':
        common_hidden=np.sort(np.random.default_rng(42).choice(common_hidden,min(50000,len(common_hidden)),replace=False))
    np.savez_compressed(destination/'common_geometry_indices.npz',wall_ids=wall_ids,inlet_ids=inlet_ids,
                        basis_centres=centres,basis_centre_sequence=basis_sequence,
                        wall_xyz=wall,inlet_xyz=inlet,inlet_values=inlet_values,
                        all_wall_xyz=P.wall_xyz,
                        wall_validation_xyz=P.wall_xyz[wall_validation_ids],
                        wall_validation_disjoint=bool(len(remaining)),
                        velocity_evaluation_ids=common_hidden,
                        inlet_cap_xyz=caps[0].points,inlet_cap_weights_m2=caps[0].weights_m2,
                        inlet_cap_normal=caps[0].outward_normal,
                        outlet_cap_xyz=caps[1].points,outlet_cap_weights_m2=caps[1].weights_m2,
                        outlet_cap_normal=caps[1].outward_normal)
    if args.prepare_only:
        assert array_hash(*[P.snaps[t].uvw for t in times])==clean_hash
        write_json(destination/'preparation_terminal.json',{'status':'prepared','case':case,'target_ms':target,
                   'conditions':list(conditions),'no_model_fitted':True,'clean_targets_unchanged':True,
                   'caps':{'inlet':caps[0].metadata(),'outlet':caps[1].metadata()}})
        print({'event':'preparation_complete','case':case,'target':target,'conditions':len(conditions)},flush=True)
        return
    integ=None
    for name,obs in conditions.items():
        if args.conditions and name not in args.conditions: continue
        for method in args.methods:
            result_path=destination/'results'/f'{name}__{method}.json'
            if result_path.exists():
                print({'event':'completed_result_retained','path':str(result_path)},flush=True)
                continue
            started=time.monotonic()
            try:
                print({'event':'fit_started','case':case,'target':target,'condition':name,'method':method},flush=True)
                if method in ('data','cont'):
                    from neural import fit_neural
                    field,fit=fit_neural(P,obs,wall,inlet,inlet_values,method,args.steps,args.init_seed,
                                         device=args.device,obs_batch=args.obs_batch,col_batch=args.col_batch)
                    predictors={t:lambda x,t=t:field.predict(x,t) for t in times}
                    import torch
                    model_path=destination/'models'/f'{name}__{method}.pt';model_path.parent.mkdir(exist_ok=True)
                    torch.save({'state_dict':field.net.state_dict(),'scales':field.scales,'fit':fit},model_path)
                else:
                    if args.rbf_family=='compact':
                        models,fit=fit_compact_pair(obs,wall,inlet,inlet_values,method=='df_rbf',
                                                   args.support_radii,args.ridges,max_iter=args.rbf_iterations,
                                                   max_nnz=args.compact_max_nnz,
                                                   centre_separation_m=args.compact_centre_separation,
                                                   wall_weight=args.rbf_wall_weight,inlet_weight=args.rbf_inlet_weight,
                                                   boundary_tolerance=args.rbf_boundary_tolerance,
                                                   column_scaling=args.compact_column_scaling,solver_device=args.rbf_solver_device,
                                                   solver_method=args.rbf_solver_method,**direct_settings(args))
                    else:
                        models,fit=fit_rbf_pair(obs,centres,wall,np.zeros((2,len(wall),3)),inlet,inlet_values,
                                               method=='df_rbf',args.length_scales,args.ridges,max_iter=args.rbf_iterations)
                    predictors={t:models[f].predict for f,t in enumerate(times)}
                    model_path=destination/'models'/f'{name}__{method}.npz';model_path.parent.mkdir(exist_ok=True)
                    if args.rbf_family=='compact':
                        np.savez_compressed(model_path,family='compact',centres=models[0].centres,
                                            coefficients=np.stack([m.coefficients for m in models]),
                                            radius_m=models[0].radius_m,vector=models[0].vector,
                                            polynomial=models[0].polynomial)
                    else:
                        np.savez_compressed(model_path,family='gaussian',centres=centres,
                                            coefficients=np.stack([m.coefficients for m in models]),
                                            ell=models[0].ell,vector=models[0].vector,origin=models[0].origin,length=models[0].length)
                predictions={t:predictors[t](P.X) for t in times} if args.score=='full' else None
                scores=diagnostic_scores(P,obs,predictors,common_hidden,P.wall_xyz,caps,predictions,(inlet,inlet_values))
                if args.score=='full':
                    postprocess=api_module('postprocess')
                    score_field=postprocess.score_field
                    if integ is None: integ=make_audited_integrator(P)
                    # Keep the ORIGINAL all-observation hidden mask for EVERY removal/operator arm.
                    solver_start=len(integ.solve_records)
                    full=score_field(P,integ,predictions,h_pre_mm=.25,h_sweep=())
                    if 'wss' in full:
                        full['wss'].update(absolute_wss_scores(full['wss'],P.snaps[target].wall_wss,P.wall_aneurysm))
                    scores.update(full,score_mode='full_historical_postprocessing',pressure_and_WSS_evaluated=True)
                    solve_records=integ.solve_records[solver_start:]
                    assert len(solve_records)==len(postprocess.PRESSURE_FORMS)
                    scores['pressure_solver']={form:record for form,record in zip(postprocess.PRESSURE_FORMS,solve_records)}
                assert array_hash(*[P.snaps[t].uvw for t in times])==clean_hash,'clean scoring targets changed'
                write_json(result_path,{'status':'complete','stage':args.stage,'case':case,'target_ms':target,
                                        'condition':name,'method':method,'observations':obs.metadata,'fit':fit,
                                        'scores':scores,'seconds':time.monotonic()-started,'clean_targets_unchanged':True,
                                        'oracle_eddy_viscosity_in_pressure':args.score=='full',
                                        'protocol_sha256':args.protocol_sha256,'model_path':str(model_path)})
            except Exception as exc:
                write_json(destination/'failures'/f'{name}__{method}.json',{'status':'failed','error':str(exc),
                           'traceback':traceback.format_exc(),'seconds':time.monotonic()-started,
                           'candidate_trials':getattr(exc,'trials',None)})
                print(traceback.format_exc(),flush=True)
                if args.fail_fast: raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',type=int,nargs='+',default=[1,9])
    parser.add_argument('--targets',type=int,nargs='+',default=[1780,2400])
    parser.add_argument('--methods',nargs='+',choices=['data','cont','scalar_rbf','df_rbf'],default=['data','cont','scalar_rbf','df_rbf'])
    parser.add_argument('--conditions',nargs='*')
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--stage',default='development_feasibility')
    parser.add_argument('--score',choices=['velocity','full'],default='full')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--centres',type=int,default=128)
    parser.add_argument('--boundary-count',type=int,default=0,help='0 uses every known wall node; subsampling is a labelled sensitivity only')
    parser.add_argument('--length-scales',type=float,nargs='+',default=[.005,.010])
    parser.add_argument('--ridges',type=float,nargs='+',default=[1e-7,1e-5,1e-3])
    parser.add_argument('--rbf-iterations',type=int,default=3000)
    parser.add_argument('--rbf-family',choices=['gaussian','compact'],default='compact')
    parser.add_argument('--support-radii',type=float,nargs='+',default=[.00375,.005,.0075])
    parser.add_argument('--compact-max-nnz',type=int,default=300_000_000)
    parser.add_argument('--compact-centre-separation',type=float,default=0.)
    parser.add_argument('--compact-column-scaling',action='store_true')
    parser.add_argument('--rbf-solver-device',default='cpu')
    parser.add_argument('--rbf-solver-method',choices=['lsmr','direct'],default='lsmr')
    add_direct_arguments(parser)
    parser.add_argument('--rbf-wall-weight',type=float,default=1.)
    parser.add_argument('--rbf-inlet-weight',type=float,default=2.)
    parser.add_argument('--rbf-boundary-tolerance',type=float,default=.02)
    parser.add_argument('--sigmas',type=float,nargs='+',default=[0.,.02])
    parser.add_argument('--noise-correlation',type=float,default=0.)
    parser.add_argument('--voxel-width-m',type=float,default=.0025)
    parser.add_argument('--geometry-margin-m',type=float,default=.0001)
    parser.add_argument('--cap-area-relative-tolerance',type=float,default=.02)
    parser.add_argument('--quadrature-tolerance',type=float,default=.01)
    parser.add_argument('--obs-batch',type=int,default=128)
    parser.add_argument('--col-batch',type=int,default=512)
    for name,default in [('grid',0),('split',42000),('noise',22001),('removal',32001),('boundary',52001),('basis',62001),('init',0)]:
        parser.add_argument('--'+name+'-seed',type=int,default=default)
    parser.add_argument('--fail-fast',action='store_true')
    args=parser.parse_args()
    if not set(args.cases)<= {1,9} or not set(args.targets)<={1780,2400}:
        parser.error('this runner is restricted to the existing development cases/windows')
    if args.steps<1 or args.centres<1 or args.boundary_count<0 or args.rbf_iterations<1:
        parser.error('positive iteration/centre counts and a nonnegative wall count are required')
    args.historical_root=args.historical_root.resolve();args.output=args.output.resolve()
    if args.output==args.historical_root or args.historical_root in args.output.parents:
        parser.error('new outputs must be OUTSIDE the historical worktree')
    sys.path.insert(0,str(args.historical_root))
    config=vars(args).copy()
    config['historical_root']=str(args.historical_root);config['output']=str(args.output)
    sources={p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')}
    config['source_hashes']=sources
    config['historical_source_hashes']={p.relative_to(args.historical_root).as_posix():file_hash(p)
        for p in (args.historical_root/'idealaorta_pinn').rglob('*.py')}
    config['software']={'python':platform.python_version(),**{n:importlib.metadata.version(n)
                          for n in ['numpy','scipy','pandas','pyarrow','torch']}}
    config['hardware']={'machine':platform.machine(),'cpu_count':os.cpu_count(),
                        'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')}
    encoded=json.dumps(config,sort_keys=True).encode()
    args.protocol_sha256=hashlib.sha256(encoded).hexdigest()
    manifest=args.output/'execution_protocol.json'
    if manifest.exists():
        existing=json.loads(manifest.read_text())
        if existing['sha256']!=args.protocol_sha256:
            raise ValueError('output root already belongs to another code/config version; choose a fresh dated directory')
    write_json(manifest,{'sha256':args.protocol_sha256,'config':config})
    for case in args.cases:
        for target in args.targets:
            run_problem(args,case,target,args.output)
    failures=list(args.output.glob('case*/failures/*.json'))
    unresolved=[str(p) for p in failures if not (p.parent.parent/'results'/p.name).exists()]
    write_json(args.output/'terminal.json',{'status':'worker_finished','protocol_sha256':args.protocol_sha256,
                                          'results':len(list(args.output.glob('case*/results/*.json'))),
                                          'failed_attempts':len(failures),'unresolved_failures':unresolved})


if __name__=='__main__':
    main()
