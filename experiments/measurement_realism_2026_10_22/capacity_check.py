"""Development-only RBF capacity checks using observed velocity and known walls.

No hidden velocity, pressure or WSS targets are read by this program. Failures
and unsettled capacity are outcomes, never a reason to drop an inconvenient arm.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from observations import Observations
from comparators import fit_rbf_pair
from run_pilot import write_json,file_hash


def evaluate(unit,output,conditions,capacities,length_scales,ridges,max_iter,
             tolerance=.02,wall_tolerance=.02):
    unit=Path(unit);output=Path(output)
    if output.exists():raise ValueError('capacity receipt already exists; retain it and choose a new output')
    gpath=unit/'common_geometry_indices.npz'
    g=np.load(gpath)
    if 'basis_centre_sequence' not in g:
        raise ValueError('capacity checks require the saved geometry sequence and independent wall pool')
    sequence=g['basis_centre_sequence']
    if max(capacities)>len(sequence):raise ValueError('requested centre count exceeds saved geometry sequence')
    receipt={'status':'running','selection_information':'no hidden targets; noisy observation hold-out and known no-slip wall only',
             'geometry_artifact_sha256':file_hash(gpath),'relative_validation_tolerance':tolerance,
             'relative_wall_rms_tolerance':wall_tolerance,'conditions':{}}
    for condition in conditions:
        path=unit/'observations'/(condition+'.npz')
        z=np.load(path)
        metadata=json.loads(path.with_suffix('.json').read_text())
        obs=Observations(tuple(z['times_ms'].tolist()),*[z[k] for k in
            ['support_m','weights','clean','values','validation','anchor_ids']],metadata)
        u=max(float(np.quantile(np.linalg.norm(obs.values[:,~obs.validation],axis=2),.995)),1e-6)
        receipt['conditions'][condition]={}
        for vector in (False,True):
            method='df_rbf' if vector else 'scalar_rbf'
            trials=[]
            for count in capacities:
                started=time.monotonic()
                try:
                    models,audit=fit_rbf_pair(obs,sequence[:count],g['wall_xyz'],np.zeros((2,len(g['wall_xyz']),3)),
                        g['inlet_xyz'],g['inlet_values'],vector,length_scales,ridges,max_iter=max_iter)
                    wall_rms=[float(np.sqrt(np.mean(np.sum(m.predict(g['wall_validation_xyz'])**2,axis=1)))) for m in models]
                    trial={'centres':count,'status':'converged','validation_error':audit['best_val_observed_rel_l2'],
                           'wall_validation_rms_m_s_per_time':wall_rms,
                           'wall_validation_rms_over_observed_U':max(wall_rms)/u,'fit_audit':audit}
                except Exception as exc:
                    trial={'centres':count,'status':'failed','error':str(exc)}
                trial['seconds']=time.monotonic()-started
                trials.append(trial)
                print({'condition':condition,'method':method,'capacity':trial},flush=True)
            converged=[t for t in trials if t['status']=='converged']
            suitable=[t for t in converged if t['wall_validation_rms_over_observed_U']<=wall_tolerance]
            selected=None
            plateau=False
            if len(converged)>=2:
                previous,last=converged[-2:]
                plateau=(previous['validation_error']-last['validation_error'])/max(last['validation_error'],1e-15)<=tolerance
            if suitable and plateau:
                best=min(t['validation_error'] for t in suitable)
                selected=min(t['centres'] for t in suitable if t['validation_error']<=(1+tolerance)*best)
            receipt['conditions'][condition][method]={'status':'capacity_settled' if selected is not None else 'capacity_unsettled',
                        'selected_centres':selected,'validation_plateau':plateau,'trials':trials}
            write_json(output,receipt)
    results=[v for methods in receipt['conditions'].values() for v in methods.values()]
    receipt['status']='capacity_settled' if all(r['status']=='capacity_settled' for r in results) else 'capacity_unsettled'
    receipt['common_conservative_centre_count']=max(r['selected_centres'] for r in results) if receipt['status']=='capacity_settled' else None
    receipt['scope']='this unit and these operators only; joint-case assessment is required before freezing final capacity'
    write_json(output,receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('unit',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--conditions',nargs='+',default=['point_all_sigma0.02','volume_core_sigma0.02'])
    p.add_argument('--capacities',nargs='+',type=int,default=[64,128,256,512,1024])
    p.add_argument('--length-scales',nargs='+',type=float,default=[.005,.010])
    p.add_argument('--ridges',nargs='+',type=float,default=[1e-5,1e-3])
    p.add_argument('--max-iter',type=int,default=1000)
    args=p.parse_args()
    if sorted(set(args.capacities))!=args.capacities:p.error('capacities must be strictly increasing')
    r=evaluate(args.unit,args.output,args.conditions,args.capacities,args.length_scales,args.ridges,args.max_iter)
    raise SystemExit(0 if r['status']=='capacity_settled' else 1)
