"""Read-only status collection for the exact v21 development jobs."""
import ast
import datetime
import json
from pathlib import Path
import subprocess


ROOT=Path('/home/olarinoyem/measurement_realism_2026_10_22')


def read(relative):
    path=ROOT/relative
    if not path.is_file():return None
    try:return json.loads(path.read_text())
    except (OSError,ValueError) as exc:return {'observation_error':str(exc)}


def events(relative):
    path=ROOT/relative
    if not path.is_file():return []
    output=[]
    for line in path.read_text(errors='replace').splitlines():
        if not line.startswith("{'event':"):continue
        try:value=ast.literal_eval(line)
        except (ValueError,SyntaxError):continue
        row={k:value[k] for k in ['event','vector','radius_m','ridge','converged','boundary_gate_pass',
             'condition','level','status','val_observed_rel_l2','shape','nnz'] if k in value}
        if value.get('direct_solver'):
            row['max_original_stationarity']=max(value['direct_solver']['original_stationarity_Frobenius_normalized'])
        if 'boundary_rms_over_U' in value:row['boundary_rms_over_U']=value['boundary_rms_over_U']
        output.append(row)
    return output


def main():
    result={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    capacity=read('compact_capacity_v21/supervisor.json');adapter=read('public_adapter_smoke_v22_running.json')
    terminal=read('public_adapter_smoke_v22_terminal.json');pids=[]
    if capacity:
        result['capacity']={k:capacity.get(k) for k in ['status','pid','started_utc','finished_utc','error']}
        result['capacity']['units']=[{k:u.get(k) for k in ['unit','status','pid','returncode','capacity_status']}
                                      for u in capacity.get('units',[])]
        pids.append(capacity['pid'])
        for unit in capacity.get('units',[]):
            if unit.get('status')=='running':pids.append(unit['pid'])
        current=next((u['unit'] for u in capacity.get('units',[]) if u.get('status')=='running'),None)
        if current:
            result['capacity']['recent_events']=events('compact_capacity_v21/'+current+'.log')[-5:]
        summaries=[]
        for path in sorted((ROOT/'compact_capacity_v21').glob('case*/level*/*.json')):
            value=json.loads(path.read_text());fit=value.get('fit',{})
            summaries.append({'path':str(path.relative_to(ROOT)),'status':value.get('status'),
                'basis_centres':value.get('basis_centres'),'separation_m':value.get('separation_m'),
                'best_val_observed_rel_l2':fit.get('best_val_observed_rel_l2'),
                'selected_candidate':fit.get('selected_candidate'),'error':value.get('error')})
        result['capacity']['completed_levels']=summaries
        result['capacity']['unit_results']=[]
        for path in sorted((ROOT/'compact_capacity_v21').glob('case*/receipt.json')):
            try:value=json.loads(path.read_text())
            except (OSError,ValueError):continue
            result['capacity']['unit_results'].append({'unit':path.parent.name,'status':value.get('status'),
                'rows':[{k:r.get(k) for k in ['condition','vector','status','capacity']} for r in value.get('results',[])]})
    if adapter:
        if not terminal:pids.extend([adapter['pid'],adapter['worker_pid']])
        result['adapter']={'supervisor_pid':adapter['pid'],'worker_pid':adapter['worker_pid'],
            'completed_results':len(list((ROOT/'public_adapter_smoke_v22').glob('draw*/results/*.json'))),
            'failure_records':len(list((ROOT/'public_adapter_smoke_v22').glob('draw*/failures/*.json'))),
            'terminal':{k:terminal.get(k) for k in ['status','returncode','assessment_returncode','assessment_status','finished_utc']}
                       if terminal else None,'assessment':read('checks/public_adapter_assessment_v22.json')}
        if result['adapter']['assessment']:
            result['adapter']['assessment']={k:result['adapter']['assessment'].get(k) for k in ['status','issues','completed_records']}
    pipeline=read('native_pipeline_v22.json')
    if pipeline:
        result['pipeline']={k:pipeline.get(k) for k in ['status','pid','started_utc','finished_utc','error','last_dependency_verification','feasibility_outcome']}
        result['pipeline']['stages']=pipeline.get('stages',[])
        if not pipeline.get('finished_utc'):
            pids.append(pipeline['pid'])
            pids.extend(stage['pid'] for stage in pipeline.get('stages',[]) if stage.get('status')=='running' and stage.get('pid'))
    tests=read('checks/integration_tests_v22_receipt.json')
    if tests:result['tests']={k:tests.get(k) for k in ['status','tests_run','passed','failed','errors','skipped','finished_utc']}
    else:
        tests=read('checks/integration_tests_v22_running.json')
        if tests:
            result['tests']={'status':'running','pid':tests['pid']};pids.append(tests['pid'])
    result['native_scoring_records']={
        'feasibility':len(list((ROOT/'native_feasibility_v22').glob('case*/results/*.json'))),
        'development':len(list((ROOT/'native_development_v22').glob('draw*/case*/results/*.json')))}
    result['independent_seal_present']=(ROOT/'independent_protocol_sealed_v22.json').exists()
    if pids:
        try:
            proc=subprocess.run(['ps','-p',','.join(map(str,sorted(set(pids)))),'-o','pid,ppid,state,etime,%cpu,rss,args'],
                                capture_output=True,text=True,timeout=5)
            result['processes']=proc.stdout
        except subprocess.TimeoutExpired:result['process_observation']='timeout; no terminal inference'
    try:
        proc=subprocess.run(['nvidia-smi','--id=1','--query-gpu=memory.used,utilization.gpu',
                             '--format=csv,noheader'],capture_output=True,text=True,timeout=5)
        result['GPU_1']=proc.stdout.strip()
    except subprocess.TimeoutExpired:result['GPU_observation']='timeout; no terminal inference'
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
