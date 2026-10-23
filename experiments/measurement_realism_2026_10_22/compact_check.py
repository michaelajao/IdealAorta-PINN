"""Check full-centre compact comparators on saved development observations only."""
from pathlib import Path
import argparse
import json
import time
import re
import numpy as np
from observations import Observations
from compact_rbf import fit_compact_pair
from direct_ridge import add_direct_arguments,direct_settings
from run_pilot import write_json,file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('unit',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--conditions',nargs='+',default=['point_all_sigma0.02','volume_core_sigma0.02'])
    p.add_argument('--support-radii',nargs='+',type=float,default=[.00375,.005,.0075])
    p.add_argument('--ridges',nargs='+',type=float,default=[1e-5,1e-3])
    p.add_argument('--max-iter',type=int,default=3000)
    p.add_argument('--max-nnz',type=int,default=300_000_000)
    p.add_argument('--wall-tolerance',type=float,default=.02)
    p.add_argument('--centre-separation',type=float,default=0.,help='geometry-only basis thinning in metres; retains every fitting boundary row')
    p.add_argument('--wall-weight',type=float,default=1.)
    p.add_argument('--inlet-weight',type=float,default=2.)
    p.add_argument('--enforce-boundary-gate',action='store_true',help='select only candidates satisfying the known wall AND inlet tolerance')
    p.add_argument('--column-scaling',action='store_true',help='right-scale the augmented regularized system without changing the objective')
    p.add_argument('--solver-device',default='cpu',help='CPU or explicit CUDA device for float64 sparse products; iteration remains SciPy LSMR')
    p.add_argument('--solver-method',choices=['lsmr','direct'],default='lsmr')
    add_direct_arguments(p)
    p.add_argument('--families',nargs='+',choices=['scalar','df'],default=['scalar','df'])
    args=p.parse_args()
    if args.output_dir.exists():raise ValueError('retain prior compact checks and choose a fresh output directory')
    args.output_dir.mkdir(parents=True)
    geom=args.unit/'common_geometry_indices.npz';g=np.load(geom)
    receipt={'status':'running','geometry_sha256':file_hash(geom),'results':[],
             'selection_information':'observed velocity hold-out; known no-slip walls; no hidden velocity/pressure/WSS read',
             'wall_rms_over_U_tolerance':args.wall_tolerance,
             'arguments':vars(args)|{'unit':str(args.unit),'output_dir':str(args.output_dir)},
             'source_hashes':{f.name:file_hash(f) for f in Path(__file__).parent.glob('*.py')}}
    match=re.fullmatch(r'case(\d+)_t(\d+)',args.unit.name)
    if not match:raise ValueError('the input must identify a development case and target time')
    case,target=map(int,match.groups())
    if case not in (1,9) or target not in (1780,2400):raise ValueError('capacity checks are restricted to development case/phase units')
    receipt['development_unit']={'case':case,'target_ms':target}
    for condition in args.conditions:
        path=args.unit/'observations'/(condition+'.npz');z=np.load(path)
        metadata=json.loads(path.with_suffix('.json').read_text())
        obs=Observations(tuple(z['times_ms'].tolist()),*[z[k] for k in
            ['support_m','weights','clean','values','validation','anchor_ids']],metadata)
        U=max(float(np.quantile(np.linalg.norm(obs.values[:,~obs.validation],axis=2),.995)),1e-6)
        for vector in (False,True):
            if ('df' if vector else 'scalar') not in args.families:continue
            name=condition+('__df' if vector else '__scalar')
            started=time.monotonic()
            try:
                models,fit=fit_compact_pair(obs,g['wall_xyz'],g['inlet_xyz'],g['inlet_values'],vector,
                    args.support_radii,args.ridges,max_iter=args.max_iter,max_nnz=args.max_nnz,
                    centre_separation_m=args.centre_separation,wall_weight=args.wall_weight,
                    boundary_tolerance=args.wall_tolerance if args.enforce_boundary_gate else None,
                    column_scaling=args.column_scaling,solver_device=args.solver_device,inlet_weight=args.inlet_weight,
                    solver_method=args.solver_method,**direct_settings(args))
                wall_rms=[float(np.sqrt(np.mean(np.sum(m.predict(g['wall_validation_xyz'])**2,axis=1)))) for m in models]
                inlet_rms=[float(np.sqrt(np.mean(np.sum((m.predict(g['inlet_xyz'])-g['inlet_values'][f])**2,axis=1))))
                           for f,m in enumerate(models)]
                row={'condition':condition,'vector':vector,'fit':fit,'wall_rms_m_s':wall_rms,
                     'observation_sha256':obs.metadata['observation_sha256'],'observation_file_sha256':file_hash(path),
                     'wall_check_disjoint_from_fitting':bool(g['wall_validation_disjoint']) if 'wall_validation_disjoint' in g else True,
                     'inlet_rms_m_s':inlet_rms,'inlet_rms_over_observed_U':max(inlet_rms)/U,
                     'wall_rms_over_observed_U':max(wall_rms)/U,
                     'status':'boundary_gate_pass' if max(wall_rms+inlet_rms)/U<=args.wall_tolerance else 'boundary_gate_failed'}
                np.savez_compressed(args.output_dir/(name+'.npz'),family='compact',centres=models[0].centres,
                    coefficients=np.stack([m.coefficients for m in models]),radius_m=models[0].radius_m,
                    vector=models[0].vector,polynomial=models[0].polynomial)
                row['model_sha256']=file_hash(args.output_dir/(name+'.npz'))
            except Exception as exc:
                row={'condition':condition,'vector':vector,'status':'fit_failed','error':str(exc)}
                if hasattr(exc,'trials'):row['trials']=exc.trials
            row['seconds']=time.monotonic()-started;receipt['results'].append(row)
            write_json(args.output_dir/(name+'.json'),row)
            write_json(args.output_dir/'receipt.json',receipt)
    receipt['status']='boundary_gates_pass' if all(r['status']=='boundary_gate_pass' for r in receipt['results']) else 'not_yet_adequate'
    receipt['scope']='these saved development observations only; additional case/phase and capacity checks remain required'
    write_json(args.output_dir/'receipt.json',receipt)
    return 0 if receipt['status']=='boundary_gates_pass' else 1


if __name__=='__main__':raise SystemExit(main())
