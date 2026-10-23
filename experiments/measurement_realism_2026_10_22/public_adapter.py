"""Verified public velocity windows and matched physical observation operators.

Development access is restricted to aorta01. Other declared release files cannot
be opened until a matching independent-protocol seal is supplied. This adapter
provides velocity targets and image-mask geometry, never pressure/WSS truth or
invented boundary conditions.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from observations import Observations,array_hash,common_validation_mask
from operators import voxel_quadrature
from voxel_adapter import VoxelDomain,LinearGridVelocity

FILES={'aorta01_HR.h5':(1,'10043cda23d60e7a3e06f519fa1e6ac9'),
       'aorta02_HR.h5':(2,'d7c63e9ea0c44957019896272c1f3a40'),
       'aorta03trans_HR.h5':(3,'6896ec247fdcd6638315ca9cd3df9165')}
WINDOWS={(9,10),(27,28),(45,46),(63,64)}


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def authorize_public_file(path,seal=None):
    """Run before ANY filesystem/HDF5 read of a declared evaluation file."""
    name=Path(path).name
    if name not in FILES:raise ValueError('file is outside the declared public release geometry set')
    if name=='aorta01_HR.h5':return 'adapter development'
    if seal is None:raise ValueError('evaluation files require a matching sealed independent protocol')
    bundle=json.loads(Path(seal).read_text(encoding='utf-8'));claimed=bundle.pop('bundle_sha256',None)
    actual=hashlib.sha256(json.dumps(bundle,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if actual!=claimed or bundle.get('status')!='sealed_design; evaluation_not_run':
        raise ValueError('independent-protocol seal is invalid')
    root=Path(__file__).parent
    for name in ['public_adapter.py','voxel_adapter.py','observations.py','operators.py','neural.py',
                 'compact_rbf.py','sparse_backend.py','direct_ridge.py','run_public_pilot.py']:
        if not (root/name).exists() or bundle.get('source_hashes',{}).get(name)!=file_hash(root/name):
            raise ValueError('independent-protocol seal does not match the current external implementation')
    return 'sealed external evaluation'


@lru_cache(maxsize=4)
def fingerprint(path,size,mtime_ns):
    source=Path(path);md5=hashlib.md5();sha=hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):md5.update(block);sha.update(block)
    after=source.stat()
    if (after.st_size,after.st_mtime_ns)!=(size,mtime_ns):raise ValueError('input changed during verification')
    return md5.hexdigest(),sha.hexdigest()


@dataclass
class PublicWindow:
    field:LinearGridVelocity
    domain:VoxelDomain
    times_ms:tuple
    metadata:dict

    @property
    def fluid_coordinates(self):
        return self.domain.origin_m+np.argwhere(self.domain.mask)*self.domain.spacing_m

    def prepare(self,draw=0):
        if draw not in (0,1,2):raise ValueError('use the three declared draws')
        centres=self.domain.cartesian_centres(.0025,draw)
        split=common_validation_mask(len(centres),42000+draw)
        core=self.domain.certify_boxes(centres,.0025,.0001);rows=np.flatnonzero(core)
        if not (core&split).any() or not (core&~split).any():raise ValueError('empty shared full-box fitting/validation stratum')
        clean=np.stack([self.field.predict(centres,f) for f in (0,1)])
        averages={}
        for order in (3,5,7):
            support,weights=voxel_quadrature(centres[core],.0025,order)
            averages[order]=np.stack([np.einsum('nqc,q->nc',
                self.field.predict(support.reshape(-1,3),f).reshape(support.shape),weights) for f in (0,1)])
            if order>3:
                errors=np.linalg.norm(averages[order-2]-averages[order],axis=(1,2))/np.linalg.norm(averages[order],axis=(1,2))
                if np.all(errors<=.01):chosen=order-2;break
        else:raise ValueError('public volume quadrature did not pass the shared 1% convergence rule')
        conditions={}
        for sigma in (0.,.02):
            point=Observations.make(self.times_ms,centres[:,None],np.ones(1),clean,split,np.arange(len(centres)),
                                    sigma,22001+draw,metadata={'operator':'continuous-grid point','draw':draw,**self.metadata})
            suffix=f'sigma{sigma:g}'
            conditions['point_all_'+suffix]=point
            conditions['point_core_'+suffix]=point.subset(rows,'matched_full_box_centres')
            support,weights=voxel_quadrature(centres[core],.0025,chosen)
            conditions['volume_core_'+suffix]=Observations.make(self.times_ms,support,weights,averages[chosen],split[core],rows,
                sigma,22001+draw,metadata={'operator':'physical full-box average in conservative mask-cell proxy','draw':draw,**self.metadata},
                realized_noise=(point.values-point.clean)[:,core])
        return conditions,{'point_centres':len(centres),'full_box_centres':len(rows),
                           'quadrature_selected_order':chosen,'quadrature_reference_order':order,
                           'quadrature_relative_difference_per_frame':errors.tolist(),
                           'quadrature_tolerance':.01,'mask_threshold':.6,'margin_m':.0001,
                           'boundary_contract':'measurements only; no wall/inlet/outlet targets',
                           'point_centres_sha256':array_hash(centres),'source':self.metadata}


def load_public_window(path,axis_audit,rows=(9,10),seal=None):
    role=authorize_public_file(path,seal)  # Must precede stat(), hash, or HDF5 access.
    import h5py
    path=Path(path);rows=tuple(rows)
    if rows not in WINDOWS:raise ValueError('window differs from the predetermined zero-based rows')
    axis_path=Path(axis_audit);axis=json.loads(axis_path.read_text(encoding='utf-8'))
    if role=='sealed external evaluation':
        bundle=json.loads(Path(seal).read_text(encoding='utf-8'))
        if bundle.get('external_adapter',{}).get('axis_audit_sha256')!=file_hash(axis_path):
            raise ValueError('component-axis calibration differs from the sealed external adapter')
    if axis.get('status')!='development_axis_calibration_pass' or not axis.get('mapping_stable_across_stencils'):
        raise ValueError('resolve the development component-axis mapping first')
    if axis.get('audit_source_sha256')!=file_hash(Path(__file__).with_name('audit_external_axes.py')):
        raise ValueError('axis calibration source differs from the proposed adapter')
    stat=path.stat();md5,sha=fingerprint(str(path.resolve()),stat.st_size,stat.st_mtime_ns)
    model,expected_md5=FILES[path.name]
    if md5!=expected_md5:raise ValueError('public input does not match its declared release MD5')
    if path.name=='aorta01_HR.h5' and sha!=axis['source_sha256']:
        raise ValueError('development axis calibration refers to a different source file')
    with h5py.File(path,'r') as f:
        labels=f['timeframe'][:]
        if not np.array_equal(labels,np.arange(1,72)) or not np.all(f['aorta_model'][:]==model):
            raise ValueError('identity/time labels differ from the frozen release contract')
        raw_mask=f['mask'][0]
        if len(f['mask'])!=1 or not np.isfinite(raw_mask).all() or raw_mask.min()<0 or raw_mask.max()>1:
            raise ValueError('mask differs from the frozen static fractional-mask contract')
        mask=raw_mask>.6
        values=np.stack([np.stack([f[name][row] for name in ('u','v','w')],axis=-1) for row in rows])
        if values.shape[1:4]!=mask.shape:raise ValueError('velocity/mask grid shapes disagree')
    spacing=np.full(3,.000594);origin=np.zeros(3)
    field=LinearGridVelocity(values,spacing,origin,np.asarray(axis['component_to_grid_axes']))
    domain=VoxelDomain(mask,spacing,origin)
    times=tuple((labels[list(rows)]*10).tolist())
    metadata={'file':path.name,'role':role,'source_md5':md5,'source_sha256':sha,'zero_based_rows':list(rows),
              'timeframe_labels':labels[list(rows)].tolist(),'times_ms':list(times),
              'axis_audit_sha256':file_hash(axis_path),'component_to_grid_axes':axis['component_to_grid_axes'],
              'adapter_source_sha256':file_hash(__file__),'voxel_adapter_source_sha256':file_hash(Path(__file__).with_name('voxel_adapter.py')),
              'coordinate_definition':'array-index axes at publication-derived isotropic 0.594 mm; origin chosen zero',
              'native_wall_geometry':False,'pressure_WSS_truth':False,'boundary_targets_available':False}
    return PublicWindow(field,domain,times,metadata)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file',type=Path,required=True);parser.add_argument('--axis-audit',type=Path,required=True)
    parser.add_argument('--rows',nargs=2,type=int,default=[9,10]);parser.add_argument('--draw',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seal',type=Path)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('retain previous public observation artifacts; choose a fresh directory')
    window=load_public_window(args.file,args.axis_audit,args.rows,args.seal)
    conditions,receipt=window.prepare(args.draw);args.output.mkdir(parents=True)
    for name,obs in conditions.items():
        np.savez_compressed(args.output/(name+'.npz'),support_m=obs.support_m,weights=obs.weights,clean=obs.clean,
                            values=obs.values,validation=obs.validation,anchor_ids=obs.anchor_ids,times_ms=obs.times_ms)
        (args.output/(name+'.json')).write_text(json.dumps(obs.metadata,indent=2),encoding='utf-8')
    (args.output/'operator_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))
