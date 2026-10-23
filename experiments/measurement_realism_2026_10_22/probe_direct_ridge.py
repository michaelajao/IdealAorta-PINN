"""Bounded scalar GPU ridge probe on saved native DEVELOPMENT observations.

Solve the SAME ||A c-y||^2 + lambda||c||^2 objective by Cholesky. Normal
equations can magnify conditioning: recompute original-system stationarity and
reject inadequate solves. This scalar, single-unit probe is not comparator
capacity evidence or a completed method comparison.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from compact_rbf import CompactRBF,full_centres
from observations import Observations
from run_pilot import write_json,file_hash


def main():
    import torch
    from scipy.sparse import vstack
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('unit',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    if args.unit.name not in {'case01_t1780','case01_t2400','case09_t1780','case09_t2400'}:
        raise ValueError('direct solve probe is development-only')
    if args.output.exists():raise ValueError('retain prior direct probes; choose a fresh output directory')
    args.output.mkdir(parents=True)
    # Independent small double-precision numerical check before the large solve.
    rng=np.random.default_rng(246);small=rng.normal(size=(40,8));target=rng.normal(size=(40,3));ridge=.003
    a=torch.tensor(small,dtype=torch.float64,device=args.device);y=torch.tensor(target,dtype=torch.float64,device=args.device)
    gram=a.T@a+ridge*torch.eye(8,dtype=torch.float64,device=args.device)
    actual=torch.cholesky_solve(a.T@y,torch.linalg.cholesky(gram)).cpu().numpy()
    expected=np.linalg.solve(small.T@small+ridge*np.eye(8),small.T@target)
    np.testing.assert_allclose(actual,expected,atol=1e-11,rtol=1e-11)
    del a,y,gram
    print({'event':'small_GPU_ridge_equivalence_pass'},flush=True)
    obs_path=args.unit/'observations/point_all_sigma0.02.npz'
    z=np.load(obs_path);metadata=json.loads(obs_path.with_suffix('.json').read_text())
    obs=Observations(tuple(z['times_ms'].tolist()),*[z[k] for k in ['support_m','weights','clean','values','validation','anchor_ids']],metadata)
    g=np.load(args.unit/'common_geometry_indices.npz');wall=g['wall_xyz'];inlet=g['inlet_xyz'];inlet_values=g['inlet_values']
    centres=full_centres(obs,wall,inlet,.001);radius=.015
    template=CompactRBF(centres,radius,False);fit=~obs.validation;validation=obs.validation
    scale=max(float(np.quantile(np.linalg.norm(obs.values[:,fit],axis=2),.995)),1e-6)
    factors=[1/np.sqrt(fit.sum()),np.sqrt(10/len(wall)),np.sqrt(10/len(inlet))]
    started=time.monotonic()
    pieces=[template.design(obs.support_m[fit],obs.weights,max_nnz=600000000),
            template.design(wall[:,None],np.ones(1),max_nnz=600000000),
            template.design(inlet[:,None],np.ones(1),max_nnz=600000000)]
    matrix=vstack([a*f for a,f in zip(pieces,factors)],format='csr');del pieces
    targets=[]
    for frame in (0,1):
        targets.append(np.concatenate([values*factor/scale for values,factor in zip(
            [obs.values[frame,fit],np.zeros((len(wall),3)),inlet_values[frame]],factors)]))
    rhs=np.concatenate(targets,axis=1);m,n=matrix.shape
    estimated_bytes=8*(m*n+3*n*n+m*6+n*6)+1024**3
    free,_=torch.cuda.mem_get_info(args.device)
    if estimated_bytes>min(12*1024**3,free-2*1024**3):raise MemoryError('direct scalar probe exceeds its explicit GPU memory bound')
    frobenius_squared=float(matrix.multiply(matrix).sum())
    print({'event':'forming_GPU_Gram','shape':matrix.shape,'nnz':matrix.nnz,'estimated_gpu_bytes':estimated_bytes},flush=True)
    matrix_gpu=torch.tensor(matrix.toarray(),dtype=torch.float64,device=args.device)
    rhs_gpu=torch.tensor(rhs,dtype=torch.float64,device=args.device)
    gram=matrix_gpu.T@matrix_gpu;normal_rhs=matrix_gpu.T@rhs_gpu
    torch.cuda.synchronize(args.device)
    setup_seconds=time.monotonic()-started
    del matrix_gpu,rhs_gpu
    receipt={'status':'running','scope':'one scalar development point-observation probe; not full comparator adequacy',
             'self_test':'GPU/NumPy ridge solutions match to 1e-11','source_sha256':file_hash(__file__),
             'kernel_source_sha256':file_hash(Path(__import__('compact_rbf').__file__)),
             'observation_file_sha256':file_hash(obs_path),'observation_sha256':metadata['observation_sha256'],
             'geometry_sha256':file_hash(args.unit/'common_geometry_indices.npz'),
             'matrix_shape':[m,n],'matrix_nnz':matrix.nnz,'setup_seconds':setup_seconds,
             'parameters':{'radius_m':radius,'centre_separation_m':.001,'wall_weight':10,'inlet_weight':10},'trials':[]}
    for ridge in (1e-6,1e-7,1e-9):
        started=time.monotonic();regularized=gram.clone();regularized.diagonal().add_(ridge)
        lower,info=torch.linalg.cholesky_ex(regularized);del regularized
        if int(info.cpu())!=0:
            row={'ridge':ridge,'status':'factorization_failed','cholesky_info':int(info.cpu())}
        else:
            solution=torch.cholesky_solve(normal_rhs,lower).cpu().numpy();del lower
            # Acceptance is checked with A and y, rather than merely a small
            # residual of the formed normal equations.
            residual=matrix@solution-rhs;gradient=matrix.T@residual+ridge*solution
            residual_norm=np.sqrt(np.sum(residual*residual,axis=0)+ridge*np.sum(solution*solution,axis=0))
            stationarity=np.linalg.norm(gradient,axis=0)/(np.sqrt(frobenius_squared+ridge*n)*residual_norm)
            models=[CompactRBF(centres,radius,False,solution[:,3*f:3*f+3]*scale) for f in (0,1)]
            predictions=np.stack([model.observe(obs.support_m[validation],obs.weights) for model in models])
            val_error=float(np.linalg.norm(predictions-obs.values[:,validation])/np.linalg.norm(obs.values[:,validation]))
            wall_errors=[float(np.sqrt(np.mean(np.sum(model.predict(wall)**2,axis=1))))/scale for model in models]
            inlet_errors=[float(np.sqrt(np.mean(np.sum((model.predict(inlet)-inlet_values[f])**2,axis=1))))/scale
                          for f,model in enumerate(models)]
            numerical=bool(np.isfinite(stationarity).all() and max(stationarity)<=1e-6)
            boundary=max(wall_errors+inlet_errors)<=.02
            row={'ridge':ridge,'status':'numerical_and_boundary_pass' if numerical and boundary else 'not_yet_adequate',
                 'original_stationarity_Frobenius_normalized':stationarity.tolist(),'numerical_gate_pass':numerical,
                 'wall_rms_over_U':wall_errors,'inlet_rms_over_U':inlet_errors,'boundary_gate_pass':boundary,
                 'val_observed_rel_l2':val_error,'regularized_objective_per_rhs':(residual_norm**2).tolist()}
            model_path=args.output/f'scalar_ridge{ridge:g}.npz'
            np.savez_compressed(model_path,centres=centres,coefficients=np.stack([model.coefficients for model in models]),
                                radius_m=radius,vector=False,polynomial=True)
            row['model_sha256']=file_hash(model_path)
        row['seconds']=time.monotonic()-started;receipt['trials'].append(row)
        write_json(args.output/'receipt.json',receipt);print({'event':'direct_scalar_trial',**row},flush=True)
    receipt['status']='direct_scalar_candidate_pass' if any(t['status']=='numerical_and_boundary_pass' for t in receipt['trials']) else 'direct_probe_complete; no_adequate_candidate'
    receipt['selection_information']='observed velocity and prescribed boundaries only; no hidden velocity/pressure/WSS scores'
    write_json(args.output/'receipt.json',receipt)


if __name__=='__main__':main()
