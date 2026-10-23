"""Disable the queued pilot without interrupting the existing capacity sweep."""
import datetime
import json
import os
from pathlib import Path
import signal
import time

root = Path('/home/olarinoyem/measurement_realism_2026_10_22')
state_path = root / 'native_pipeline_v22_threads8.json'
receipt_path = root / 'checks/user_audit_pause_20261023.json'
state = json.loads(state_path.read_text())
pid = state['pid']
proc = Path('/proc') / str(pid)
command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode() if proc.exists() else ''
if command:
    assert command.endswith('run_native_pipeline_v22_threads8.py '), command
    assert state['status'] == 'waiting_for_verified_development_dependencies'
    assert state['stages'] == [], 'Do not interrupt an unexpected active stage'
    os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        if not proc.exists() or (proc / 'stat').read_text().split(') ', 1)[1].split()[0] == 'Z':
            break
        time.sleep(.2)
    else:
        raise RuntimeError('Controller has not stopped; no stronger signal sent')
terminal = json.loads(state_path.read_text())
receipt = {
    'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'reason': 'User supplied audit requests pausing goal and redesign before full pilot',
    'controller_pid': pid,
    'controller_terminal_state': terminal,
    'automatic_feasibility_and_final_pilot_launch': 'disabled; controller stopped',
    'capacity_sweep': 'left running unchanged; supervisor 1780664 and current worker 1780672',
    'new_fits_or_benchmarks_launched': False,
    'independent_evaluation_launched': False,
}
assert terminal['stages'] == []
assert terminal.get('finished_utc'), terminal
receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
