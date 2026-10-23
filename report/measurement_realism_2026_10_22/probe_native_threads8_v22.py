"""Same saved native candidate at eight threads; no accuracy tuning or targets."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import traceback

ROOT=Path('/home/olarinoyem/measurement_realism_2026_10_22')
SOURCE=ROOT/'code_v22'
OUTPUT=ROOT/'native_threads8_probe_v22'


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker():
    import faulthandler
    faulthandler.enable(all_threads=True)
    sys.path[:0]=[str(SOURCE),'/home/olarinoyem/aorta_rescore']
    import numpy as np
    import torch
    from observations import Observations,array_hash
    from compact_rbf import fit_compact_pair
    assert torch.get_num_threads()==8,torch.get_num_threads()
    tests=json.loads((ROOT/'checks/integration_tests_v22_receipt.json').read_text())
    for name,digest in tests['source_hashes'].items():assert sha(SOURCE/name)==digest,name
    unit=ROOT/'preparation_v5/case01_t1780'
    path=unit/'observations/point_all_sigma0.02.npz'
    with np.load(path) as z:values={k:z[k].copy() for k in z.files}
    metadata=json.loads(path.with_suffix('.json').read_text())
    assert array_hash(*[values[k] for k in ['support_m','weights','values','validation','anchor_ids']])==metadata['observation_sha256']
    obs=Observations(tuple(values['times_ms'].tolist()),*[values[k] for k in
        ['support_m','weights','clean','values','validation','anchor_ids']],metadata)
    for name in ['support_m','weights','clean','values','validation','anchor_ids']:getattr(obs,name).flags.writeable=False
    with np.load(unit/'common_geometry_indices.npz') as z:g={k:z[k].copy() for k in z.files}
    original=ROOT/'compact_capacity_v21/case01_t1780/level00/point_all_sigma0.02__df.json'
    old=json.loads(original.read_text())
    previous=next(t for t in old['fit']['trials'] if t['radius_m']==.02 and t['ridge']==1e-9)
    models,fit=fit_compact_pair(obs,g['wall_xyz'],g['inlet_xyz'],g['inlet_values'],True,
        [.02],[1e-9],max_iter=20000,max_nnz=1200000000,centre_separation_m=.001,
        wall_weight=10.,inlet_weight=10.,boundary_tolerance=.02,
        solver_device='cuda:0',solver_method='direct',direct_factor_device='cpu',
        direct_cpu_factor_backend='torch',direct_block_columns=512,
        direct_max_gpu_gib=42.,direct_max_host_gib=96.)
    predicted=np.stack([m.observe(obs.support_m[obs.validation],obs.weights) for m in models])
    holdout=original.with_name('point_all_sigma0.02__df_holdout.npz')
    assert sha(holdout)==old['holdout_sha256']
    with np.load(holdout) as z:
        assert np.array_equal(z['anchor_ids'],obs.anchor_ids[obs.validation])
        assert np.array_equal(z['observed'],obs.values[:,obs.validation])
        prior=z['predicted'].copy()
    difference=float(np.max(np.abs(predicted-prior)))
    np.testing.assert_allclose(predicted,prior,rtol=1e-7,atol=1e-8)
    trial=fit['trials'][0]
    assert trial['matrix_shape']==previous['matrix_shape'] and trial['matrix_nnz']==previous['matrix_nnz']
    assert trial['converged'] and trial['boundary_gate_pass']
    assert abs(trial['val_observed_rel_l2']-previous['val_observed_rel_l2'])<1e-7
    np.savez_compressed(OUTPUT/'model.npz',centres=models[0].centres,
        coefficients=np.stack([m.coefficients for m in models]),radius_m=models[0].radius_m,
        vector=True,polynomial=True)
    np.savez_compressed(OUTPUT/'holdout.npz',predicted=predicted,anchor_ids=obs.anchor_ids[obs.validation])
    (OUTPUT/'fit.json').write_text(json.dumps(fit,indent=2)+'\n')
    result={'status':'matched_native_thread_candidate_pass','finished_utc':now(),
        'scope':'one matched native Case 1 systolic noisy-point vector candidate on shared machine; not capacity or universal speedup',
        'source_snapshot':'code_v22','source_hashes':tests['source_hashes'],
        'observation_sha256':metadata['observation_sha256'],'old_level_receipt_sha256':sha(original),
        'thread_counts':[4,8],'matrix_shape':trial['matrix_shape'],'matrix_nnz':trial['matrix_nnz'],
        'maximum_heldout_prediction_difference_m_s':difference,
        'observed_validation_absolute_difference':abs(trial['val_observed_rel_l2']-previous['val_observed_rel_l2']),
        'factor_and_solve_seconds_4_threads':previous['direct_solver']['seconds'],
        'factor_and_solve_seconds_8_threads':trial['direct_solver']['seconds'],
        'factor_and_solve_elapsed_ratio_4_to_8':previous['direct_solver']['seconds']/trial['direct_solver']['seconds'],
        'boundary_gate_pass':trial['boundary_gate_pass'],
        'maximum_original_stationarity':max(trial['direct_solver']['original_stationarity_Frobenius_normalized']),
        'known_boundary_rms_over_U':trial['boundary_rms_over_U'],
        'model_sha256':sha(OUTPUT/'model.npz'),'holdout_sha256':sha(OUTPUT/'holdout.npz'),
        'hidden_velocity_pressure_WSS_targets_read':False,'evaluation_arrays_accessed':False}
    (OUTPUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    assert 'brosnan' in socket.gethostname().lower()
    if '--worker' in sys.argv:return worker()
    assert not OUTPUT.exists(),'retain earlier probe'
    OUTPUT.mkdir()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OPENBLAS_NUM_THREADS='8',
        OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONPATH='/home/olarinoyem/aorta_rescore',
        PYTHONPYCACHEPREFIX='/tmp/idealaorta_pilot_pycache_olarinoyem')
    value={'status':'running','pid':os.getpid(),'started_utc':now(),'source_snapshot':str(SOURCE),
        'supervisor_sha256':sha(__file__),'environment':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}}
    with (OUTPUT/'worker.log').open('x') as log:
        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker'],cwd=SOURCE,
            env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        value['worker_pid']=child.pid
        (OUTPUT/'running.json').write_text(json.dumps(value,indent=2)+'\n')
        try:value['returncode']=child.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            value['timeout']=True;os.killpg(child.pid,signal.SIGTERM)
            try:value['returncode']=child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);value['returncode']=child.wait()
    value.update(status='finished' if value['returncode']==0 else 'probe_failed',finished_utc=now())
    (OUTPUT/'terminal.json').write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':main()
