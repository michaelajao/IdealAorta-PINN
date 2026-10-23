"""One observed-data vector candidate with native crash diagnostics; no targets."""
import datetime
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT=Path('/home/olarinoyem/measurement_realism_2026_10_22')
SOURCE=ROOT/'code_v20'


def worker():
    faulthandler.enable(all_threads=True)
    faulthandler.dump_traceback_later(300,repeat=True)
    sys.path.insert(0,str(SOURCE))
    import numpy as np
    from observations import Observations
    from compact_rbf import fit_compact_pair
    checks=json.loads((ROOT/'checks/integration_tests_v20_receipt.json').read_text())
    for name,digest in checks['source_hashes'].items():
        assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==digest,name
    unit=ROOT/'preparation_v5/case01_t1780'
    z=np.load(unit/'observations/point_all_sigma0.02.npz')
    metadata=json.loads((unit/'observations/point_all_sigma0.02.json').read_text())
    obs=Observations(tuple(z['times_ms'].tolist()),*[z[k] for k in
        ['support_m','weights','clean','values','validation','anchor_ids']],metadata)
    g=np.load(unit/'common_geometry_indices.npz')
    def trace(frame,event,arg):
        if event=='line' and frame.f_code.co_filename.endswith('/direct_ridge.py'):
            print({'diagnostic_line':frame.f_lineno,'function':frame.f_code.co_name,
                   'monotonic':time.monotonic()},flush=True)
        return trace
    sys.settrace(trace)
    models,fit=fit_compact_pair(obs,g['wall_xyz'],g['inlet_xyz'],g['inlet_values'],True,
        [.015],[1e-9],max_iter=20000,max_nnz=1200000000,centre_separation_m=.001,
        wall_weight=10.,inlet_weight=10.,boundary_tolerance=.02,
        solver_device='cuda:0',solver_method='direct',direct_factor_device='cpu',
        direct_block_columns=512,direct_max_gpu_gib=42.,direct_max_host_gib=160.)
    sys.settrace(None)
    (ROOT/'checks/native_vector_diagnostic_v20_fit.json').write_text(json.dumps(fit,indent=2)+'\n')


def main():
    assert 'brosnan' in socket.gethostname().lower()
    if '--worker' in sys.argv:return worker()
    destination=ROOT/'checks/native_vector_diagnostic_v20_terminal.json'
    assert not destination.exists(),'preserve previous diagnostic'
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',
        PYTHONPATH='/home/olarinoyem/aorta_rescore',
        PYTHONPYCACHEPREFIX='/tmp/idealaorta_pilot_pycache_olarinoyem',
        PYTHONFAULTHANDLER='1',CUDA_LAUNCH_BLOCKING='1')
    result={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'supervisor_pid':os.getpid(),'source':'code_v20',
        'scope':'one observed-only native vector candidate; diagnostic, not capacity proof'}
    with (ROOT/'checks/native_vector_diagnostic_v20.log').open('x') as log:
        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker'],
            cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True)
        result['worker_pid']=child.pid
        (ROOT/'checks/native_vector_diagnostic_v20_running.json').write_text(json.dumps(result,indent=2)+'\n')
        try:result['returncode']=child.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            result['timeout']=True
            os.killpg(child.pid,signal.SIGTERM)
            result['returncode']=child.wait(timeout=30)
    result['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    destination.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
