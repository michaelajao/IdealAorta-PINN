"""Calibrate stored velocity components using ONLY aorta01 development fields.

No evaluation-file argument exists. A signed permutation is ranked by the
divergence/gradient-energy ratio on interior geometry-only stencils. A global
velocity-sign ambiguity remains; fix the first mapped sign positive. This is a
canonical numerical convention, not identification of anatomical directions.
Both one- and two-grid-step stencils must identify the same well-separated map.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import itertools
import json
import numpy as np


def rank_axis_maps(derivatives):
    """Input H[n,i,c] = derivative of STORED component c along grid axis i."""
    h=np.asarray(derivatives,float)
    if h.ndim!=3 or h.shape[1:]!=(3,3) or not len(h) or not np.isfinite(h).all():
        raise ValueError('finite nonempty derivative blocks (N,3,3) required')
    energy=float(np.sum(h*h))
    if energy<=np.finfo(float).tiny:raise ValueError('no usable gradient energy for axis calibration')
    rows=[]
    for permutation in itertools.permutations(range(3)):
        for last_signs in itertools.product((-1,1),repeat=2):
            matrix=np.zeros((3,3));matrix[np.arange(3),permutation]=(1,*last_signs)
            divergence=np.einsum('nic,ic->n',h,matrix)
            rows.append({'component_to_grid_axes':matrix.astype(int).tolist(),
                         'divergence_over_gradient_Frobenius':float(np.sqrt(np.dot(divergence,divergence)/energy))})
    return sorted(rows,key=lambda row:row['divergence_over_gradient_Frobenius'])


def audit(root,output):
    import h5py
    from scipy.ndimage import binary_erosion
    root=Path(root);output=Path(output)
    if output.exists():raise ValueError('retain prior axis audits; choose a fresh output file')
    source=root/'aorta01_HR.h5';download=json.loads((root/'aorta01_download.json').read_text())
    if download.get('status')!='verified_download' or download.get('md5')!='10043cda23d60e7a3e06f519fa1e6ac9':
        raise ValueError('complete and verify the declared development download before analysing its velocities')
    sha=hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):sha.update(block)
    if sha.hexdigest()!=download['sha256']:raise ValueError('development file no longer matches the verified download')
    frames=[9,10,27,28,45,46,63,64];spacing=.000594
    blocks={1:[],2:[]};per_frame={};maxima=[]
    with h5py.File(source,'r') as f:
        if not np.array_equal(f['timeframe'][:],np.arange(1,72)) or not np.all(f['aorta_model'][:]==1):
            raise ValueError('development identity/time indexing differs from the inspected contract')
        mask=f['mask'][0]>.9
        core=binary_erosion(mask,structure=np.ones((3,3,3),bool),iterations=2,border_value=0)
        coords=np.argwhere(core)
        if len(coords)<100:raise ValueError('insufficient geometry-only interior stencils')
        ids=np.sort(np.random.default_rng(72001).choice(len(coords),min(50000,len(coords)),replace=False))
        coords=coords[ids]
        for frame in frames:
            fields=np.stack([f[name][frame] for name in ('u','v','w')],axis=-1)
            if not np.isfinite(fields).all():raise ValueError('nonfinite development velocities')
            maxima.append({'row':frame,'timeframe_label':int(f['timeframe'][frame]),
                           'max_abs_stored_component':np.max(np.abs(fields),axis=(0,1,2)).tolist(),
                           'u_v_w_max_metadata':[float(f[name][frame]) for name in ('u_max','v_max','w_max')]})
            per_frame[str(frame)]={}
            for step in (1,2):
                derivative=np.empty((len(coords),3,3))
                for axis in range(3):
                    delta=np.eye(3,dtype=int)[axis]*step
                    plus=coords+delta;minus=coords-delta
                    derivative[:,axis]=(fields[tuple(plus.T)]-fields[tuple(minus.T)])/(2*spacing*step)
                blocks[step].append(derivative)
                per_frame[str(frame)][str(step)]=rank_axis_maps(derivative)[:2]
            print({'event':'development_axis_stencils','row':frame,'points':len(coords)},flush=True)
    ranking={str(step):rank_axis_maps(np.concatenate(parts)) for step,parts in blocks.items()}
    best=[ranking[str(step)][0] for step in (1,2)]
    stable=best[0]['component_to_grid_axes']==best[1]['component_to_grid_axes']
    gaps=[ranking[str(step)][1]['divergence_over_gradient_Frobenius']/max(ranking[str(step)][0]['divergence_over_gradient_Frobenius'],1e-15)
          for step in (1,2)]
    accepted=stable and min(gaps)>=2 and max(row['divergence_over_gradient_Frobenius'] for row in best)<=.2
    result={'status':'development_axis_calibration_pass' if accepted else 'axis_mapping_unresolved',
            'audited_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'file_role':'aorta01 adapter development only','source_sha256':sha.hexdigest(),
            'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'physical_spacing_m_assumption':[spacing]*3,'spacing_source':'publication; HDF5 carries no affine or spacing',
            'component_units_source':'upstream preparation code uses u/v/w and u_max/v_max/w_max with VENC in m/s',
            'geometry_stencil':'mask > 0.9, two 3x3x3 erosions; same <=50000 points at both steps/all frames',
            'sample_seed':72001,'sample_points':len(coords),'zero_based_rows':frames,
            'ranking':ranking,'per_frame_top_two':per_frame,'component_maxima':maxima,
            'mapping_stable_across_stencils':stable,'runner_up_to_best_ratios':gaps,
            'acceptance_rule':'same aggregate map at steps 1/2, runner-up/best >=2, normalized divergence <=0.2',
            'component_to_grid_axes':best[0]['component_to_grid_axes'] if accepted else None,
            'global_sign_convention':'first mapped component positive; absolute anatomical/flow orientation is not established',
            'evaluation_files_accessed':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.development_root,args.output)
    print(json.dumps({k:result[k] for k in ('status','component_to_grid_axes','runner_up_to_best_ratios')},indent=2))
    raise SystemExit(0 if result['status']=='development_axis_calibration_pass' else 1)
