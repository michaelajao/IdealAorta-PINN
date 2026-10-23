"""Bounded native pilot and protocol seal; never downloads evaluation fields.

Waits for the ALREADY RUNNING v21 capacity sweep and the v22 adapter assessment.
The numerical comparator is unchanged. Missing/live dependencies are never
restarted by this controller. All outputs and failed attempts are retained.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT=Path('/home/olarinoyem/measurement_realism_2026_10_22')
SOURCE=ROOT/'code_v22'
INCOMING=ROOT/'incoming_v22_20261023'
PYTHON='/home/olarinoyem/miniconda3/envs/deep_tf/bin/python'


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def alive(pid,expected):
    path=Path('/proc')/str(pid)
    try:
        command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        state=(path/'stat').read_text().split(') ',1)[1].split()[0]
    except FileNotFoundError:return False
    return expected in command and state!='Z'


def main():
    assert 'brosnan' in socket.gethostname().lower()
    plan_path=INCOMING/'native_pipeline_plan_v22.json'
    plan=json.loads(plan_path.read_text())
    if '--print-plan' in sys.argv:
        print(json.dumps(plan,indent=2));return 0
    destination=ROOT/'native_pipeline_v22.json'
    if destination.exists():raise RuntimeError('retain prior controller; never launch it twice')
    tests=json.loads((ROOT/'checks/integration_tests_v22_receipt.json').read_text())
    assert tests['returncode']==0 and tests['tests_run']==66 and not tests['skipped']
    sys.path[:0]=[str(SOURCE),'/home/olarinoyem/aorta_rescore']
    sys.pycache_prefix='/tmp/idealaorta_pilot_pycache_olarinoyem'
    from seal_protocol import validate_capacity,validate_external_contract,seal
    from assess_pilot import assess,assess_single_draw
    def source_guard():
        for name,digest in tests['source_hashes'].items():
            if sha(SOURCE/name)!=digest:raise RuntimeError('tested source changed: '+name)
        if sha(SOURCE/'independent_protocol.json')!=plan['independent_design_sha256']:
            raise RuntimeError('external design changed after pipeline declaration')
        if sha(INCOMING/'prior_access_declaration_20261023.json')!=plan['access_declaration_sha256']:
            raise RuntimeError('access declaration changed after pipeline declaration')
    source_guard()
    # These seven files define the saved capacity objective, observations and
    # plateau audit. Reporting/assessment additions do not invalidate their fits.
    for name,digest in plan['capacity_compatible_source_hashes'].items():
        if sha(SOURCE/name)!=digest:raise RuntimeError('capacity implementation differs: '+name)
    env=os.environ.copy();env.update(plan['environment'])
    state={'status':'waiting_for_verified_development_dependencies','pid':os.getpid(),
        'started_utc':now(),'controller_sha256':sha(__file__),'plan_sha256':sha(plan_path),
        'source_snapshot':str(SOURCE),'source_hashes':tests['source_hashes'],
        'scope':'native development pilot then independent design seal; no external evaluation execution',
        'stages':[]}
    write(destination,state);child=None
    def terminate(signum,frame):raise KeyboardInterrupt('native pipeline termination requested')
    signal.signal(signal.SIGTERM,terminate)
    def run(command,label,timeout):
        nonlocal child
        source_guard()
        if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('less than declared 8 GiB disk reserve')
        item={'label':label,'command':command,'started_utc':now(),'timeout_seconds':timeout,'status':'running'}
        state['stages'].append(item);state['status']='running_'+label
        with (ROOT/'checks'/('native_pipeline_v22_'+label+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            item['pid']=child.pid;write(destination,state)
            try:code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                item['timeout']=True;os.killpg(child.pid,signal.SIGTERM)
                try:code=child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        child=None;item.update(returncode=code,finished_utc=now(),status='finished' if code==0 else 'execution_failed')
        write(destination,state)
        if code:raise RuntimeError(label+' did not finish successfully; retain outputs')
    try:
        deadline=time.monotonic()+plan['dependency_wait_timeout_seconds']
        while True:
            capacity=json.loads((ROOT/'compact_capacity_v21/supervisor.json').read_text())
            adapter_path=ROOT/'public_adapter_smoke_v22_terminal.json'
            adapter=json.loads(adapter_path.read_text()) if adapter_path.exists() else None
            if capacity.get('finished_utc') and capacity['status']!='all_four_units_capacity_settled':
                raise RuntimeError('capacity dependency finished without all four settled units; no scored pilot launched')
            if adapter and adapter.get('status')!='adapter_checks_pass':
                raise RuntimeError('adapter dependency finished without passing current assessment')
            if capacity.get('finished_utc') and adapter:break
            if not capacity.get('finished_utc') and not alive(capacity['pid'],'run_capacity_v21_20261023.py'):
                raise RuntimeError('capacity process missing; never restart it from this controller')
            if not adapter:
                handle=ROOT/'public_adapter_smoke_v22_running.json'
                if not handle.exists():raise RuntimeError('v22 adapter must be launched and verified first')
                adapter_handle=json.loads(handle.read_text())
                if not alive(adapter_handle['pid'],'run_adapter_v22_20261023.py'):
                    raise RuntimeError('adapter process missing; never restart it from this controller')
            state['last_dependency_verification']={'observed_utc':now(),'capacity_pid':capacity['pid'],
                'capacity_finished':bool(capacity.get('finished_utc')),'adapter_finished':bool(adapter)}
            write(destination,state)
            if time.monotonic()>deadline:raise RuntimeError('bounded dependency observation elapsed; no restart or target access')
            time.sleep(30)
        capacity_paths=[ROOT/'compact_capacity_v21'/unit/'receipt.json' for unit in plan['native_units']]
        for path in capacity_paths:validate_capacity(json.loads(path.read_text()),SOURCE,path.parent)
        assessment=json.loads((ROOT/'checks/public_adapter_assessment_v22.json').read_text())
        validate_external_contract(json.loads((SOURCE/'independent_protocol.json').read_text()),assessment,SOURCE)
        state['capacity_receipts']=[{'path':str(p),'sha256':sha(p)} for p in capacity_paths]
        write(destination,state)
        for job in plan['pilot_commands']:
            output=Path(job['output'])
            if output.exists():raise RuntimeError('fresh pilot output required; do not overwrite '+str(output))
            run([PYTHON,'-u']+job['argv'],job['label'],job['timeout_seconds'])
            audit=assess(output,'feasibility') if job['audit_stage']=='feasibility' else assess_single_draw(output,20000)
            write(ROOT/'checks'/('native_pipeline_v22_'+job['label']+'_audit.json'),audit)
            if job['audit_stage']=='feasibility':
                # A short fit may reveal neural boundary inadequacy. Retain that
                # finding, then execute the ALREADY DECLARED final 20k budget.
                # Missing records, comparator failures, information mismatches,
                # or unconverged pressure solves still stop the pipeline.
                allowed=all(('/data: known wall/inlet numerical adequacy' in x or
                             '/cont: known wall/inlet numerical adequacy' in x) for x in audit['issues'])
                if audit['missing_results'] or audit['completed_records']!=128 or not allowed:
                    raise RuntimeError('bounded feasibility coverage/information/scoring gate failed')
                state['feasibility_outcome']='all_128_coverage_and_contracts_pass' if not audit['issues'] else 'all_128_present_neural_boundary_inadequacy_retained_proceed_fixed_20k_budget'
                state['feasibility_issues_retained']=audit['issues'];write(destination,state)
        development=ROOT/'native_development_v22'
        audit=assess(development,'development')
        write(ROOT/'checks/native_development_v22_audit.json',audit)
        if audit['status']!='development_pilot_coverage_and_contracts_pass' or audit['completed_records']!=384:
            raise RuntimeError('full 384-record development audit failed; independent design remains unsealed')
        run([PYTHON,'-u',str(INCOMING/'summarize_native_pilot_v22.py'),str(development)],'report',1800)
        source_guard()
        bundle=seal(development,capacity_paths,INCOMING/'prior_access_declaration_20261023.json',
                    ROOT/'independent_protocol_sealed_v22.json',ROOT/'checks/public_adapter_assessment_v22.json')
        state.update(status='native_pilot_complete_independent_design_sealed_external_evaluation_not_run',
                     sealed_bundle_sha256=bundle['bundle_sha256'])
    except BaseException:
        state['status']='stopped_at_development_gate_or_execution_error';state['error']=traceback.format_exc()
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    finally:
        state['finished_utc']=now();write(destination,state)
    return 0 if state['status'].startswith('native_pilot_complete_') else 1


if __name__=='__main__':sys.exit(main())
