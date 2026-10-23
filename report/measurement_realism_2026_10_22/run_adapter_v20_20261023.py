"""CPU-only, aorta01 DEVELOPMENT integration check for the current source.

Eight short point/volume fits, followed by the adapter assessment. This does
not access evaluation geometries or establish accuracy/pressure/WSS results.
"""
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
SOURCE=ROOT/'code_v20'
PYTHON='/home/olarinoyem/miniconda3/envs/deep_tf/bin/python'


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    if 'brosnan' not in socket.gethostname().lower():raise RuntimeError('Brosnan-only execution')
    tests=json.loads((ROOT/'checks/integration_tests_v20_receipt.json').read_text())
    if tests.get('returncode')!=0 or tests.get('tests_run')!=63 or tests.get('skipped'):
        raise RuntimeError('complete current integration checks must pass first')
    for name,digest in tests['source_hashes'].items():
        if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('source changed')
    output=ROOT/'public_adapter_smoke_v20';assessment=ROOT/'checks/public_adapter_assessment_v20.json'
    terminal=ROOT/'public_adapter_smoke_v20_terminal.json'
    if output.exists() or terminal.exists() or assessment.exists():raise RuntimeError('retain previous integration outputs')
    axis=ROOT/'external_adapter_development/aorta01_axis_audit_v11.json'
    command=[PYTHON,'-u','run_public_pilot.py','--historical-root','/home/olarinoyem/aorta_rescore',
        '--file',str(ROOT/'external_adapter_development/aorta01_HR.h5'),'--axis-audit',str(axis),
        '--output',str(output),'--rows','9','10','--draws','0','--steps','100','--device','cpu',
        '--conditions','point_core_sigma0','volume_core_sigma0','--support-radii','.015','--ridges','.001',
        '--rbf-iterations','2000','--rbf-max-nnz','600000000','--centre-separation','.001',
        '--solver-device','cpu','--solver-method','direct','--direct-factor-device','cpu',
        '--direct-block-columns','512','--direct-max-host-gib','24','--max-score-nodes','5000']
    assess=[PYTHON,'-u','assess_public_adapter.py',str(output),'--axis-audit',str(axis),'--output',str(assessment)]
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',
        CUDA_VISIBLE_DEVICES='',PYTHONPATH='/home/olarinoyem/aorta_rescore',
        PYTHONPYCACHEPREFIX='/tmp/idealaorta_pilot_pycache_olarinoyem')
    state={'status':'running','pid':os.getpid(),'started_utc':now(),'scope':'aorta01 development adapter integration only',
           'command':command,'assessment_command':assess,'source_snapshot':str(SOURCE),
           'supervisor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'environment':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']}}
    child=None
    def terminate(signum,frame):raise KeyboardInterrupt('adapter supervisor termination requested')
    signal.signal(signal.SIGTERM,terminate)
    try:
        with (ROOT/'public_adapter_smoke_v20.log').open('x') as log:
            child=subprocess.Popen(command,cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['worker_pid']=child.pid
            (ROOT/'public_adapter_smoke_v20_running.json').write_text(json.dumps(state,indent=2)+'\n')
            try:state['returncode']=child.wait(timeout=3*3600)
            except subprocess.TimeoutExpired:
                state['timeout']=True;child.terminate()
                try:state['returncode']=child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();state['returncode']=child.wait()
        child=None
        with (ROOT/'checks/public_adapter_assessment_v20.log').open('x') as log:
            result=subprocess.run(assess,cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,stdout=log,
                                  stderr=subprocess.STDOUT,timeout=1800)
            state['assessment_returncode']=result.returncode
        if assessment.exists():state['assessment_status']=json.loads(assessment.read_text()).get('status')
        state['status']='adapter_checks_pass' if state['returncode']==state['assessment_returncode']==0 else 'adapter_checks_incomplete_or_failed'
    except BaseException:
        state['status']='supervisor_interrupted_or_failed';state['error']=traceback.format_exc()
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    finally:
        state['finished_utc']=now();terminal.write_text(json.dumps(state,indent=2)+'\n')
    return 0 if state['status']=='adapter_checks_pass' else 1


if __name__=='__main__':sys.exit(main())
