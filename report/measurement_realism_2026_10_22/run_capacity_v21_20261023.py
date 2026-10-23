"""Brosnan-only supervisor for the four declared native development units.

Run after source deployment and all current integration checks pass. Each
capacity process is serial, uses GPU 1 for Gram products and four CPU threads,
and has an eight-hour execution bound. Scientific failures remain recorded.
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
SOURCE=ROOT/'code_v21'
OUTPUT=ROOT/'compact_capacity_v21'
PYTHON='/home/olarinoyem/miniconda3/envs/deep_tf/bin/python'


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    if 'brosnan' not in socket.gethostname().lower():raise RuntimeError('Brosnan-only execution')
    checks=json.loads((ROOT/'checks/integration_tests_v21_receipt.json').read_text())
    if checks.get('returncode')!=0 or checks.get('tests_run')!=64 or checks.get('skipped'):
        raise RuntimeError('complete current scientific integration suite must pass first')
    for name,digest in checks['source_hashes'].items():
        if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('checked source changed: '+name)
    plan=json.loads((ROOT/'incoming_v21_20261023/resume_v21.json').read_text())
    if OUTPUT.exists():raise RuntimeError('retain prior outputs; never launch this stage twice')
    free_gpu=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free',
                                        '--format=csv,noheader,nounits'],text=True).strip())
    if free_gpu<40*1024:raise RuntimeError('GPU 1 lacks the declared free-memory reserve')
    with open('/proc/meminfo') as stream:
        available=next(int(line.split()[1])*1024 for line in stream if line.startswith('MemAvailable:'))
    if available<170*1024**3:raise RuntimeError('insufficient available host memory for this stage')
    OUTPUT.mkdir()
    env=os.environ.copy();env.update(plan['environment'])
    state={'status':'running','pid':os.getpid(),'started_utc':now(),'source_snapshot':str(SOURCE),
           'supervisor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'integration_receipt_sha256':hashlib.sha256((ROOT/'checks/integration_tests_v21_receipt.json').read_bytes()).hexdigest(),
           'source_hashes':checks['source_hashes'],'unit_timeout_seconds':8*3600,
           'execution_policy':'four native case/phase units serial; GPU 1; four CPU threads; retain all outcomes',
           'scope':'observed-data comparator capacity; not the full scored pilot or independent evaluation',
           'preflight':{'GPU_1_free_MiB':free_gpu,'available_host_bytes':available},'units':[]}
    state_path=OUTPUT/'supervisor.json';write(state_path,state)
    def terminate(signum,frame):raise KeyboardInterrupt('supervisor termination requested')
    signal.signal(signal.SIGTERM,terminate)
    child=None
    try:
        for job in plan['capacity_commands']:
            command=[PYTHON,'-u']+job['argv']
            item={'unit':job['unit'],'command':command,'started_utc':now(),'status':'running'}
            state['units'].append(item)
            with (OUTPUT/(job['unit']+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,
                                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                item['pid']=child.pid;write(state_path,state)
                try:code=child.wait(timeout=8*3600)
                except subprocess.TimeoutExpired:
                    item['timeout']=True;child.terminate()
                    try:code=child.wait(timeout=30)
                    except subprocess.TimeoutExpired:child.kill();code=child.wait()
            child=None;item.update(returncode=code,finished_utc=now())
            receipt=OUTPUT/job['unit']/'receipt.json'
            if receipt.is_file():
                value=json.loads(receipt.read_text());item['capacity_status']=value.get('status')
                item['capacity_receipt_sha256']=hashlib.sha256(receipt.read_bytes()).hexdigest()
            item['status']='capacity_settled' if code==0 and item.get('capacity_status')=='capacity_settled' else 'capacity_unsettled_or_execution_failed'
            write(state_path,state)
        state['status']='all_four_units_capacity_settled' if all(v['status']=='capacity_settled' for v in state['units']) else 'capacity_remains_unsettled'
    except BaseException:
        state['status']='supervisor_interrupted_or_failed';state['error']=traceback.format_exc()
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    finally:
        state['finished_utc']=now();write(state_path,state)
    return 0 if state['status']=='all_four_units_capacity_settled' else 1


if __name__=='__main__':sys.exit(main())
