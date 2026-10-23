"""Complete source-matched checks on Brosnan; no evaluation data access."""
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import traceback
import unittest

ROOT=Path('/home/olarinoyem/measurement_realism_2026_10_22')
SOURCE=ROOT/'code_v21'


def main():
    assert 'brosnan' in socket.gethostname().lower()
    os.environ.update(CUDA_VISIBLE_DEVICES='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    sys.pycache_prefix='/tmp/idealaorta_pilot_pycache_olarinoyem'
    os.chdir(SOURCE)
    sys.path[:0]=[str(SOURCE),'/home/olarinoyem/aorta_rescore']
    receipt=ROOT/'checks/integration_tests_v21_receipt.json'
    assert not receipt.exists(),'retain previous checks'
    value={'pid':os.getpid(),'host':socket.gethostname(),
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_snapshot':str(SOURCE),
        'scope':'complete scientific suite, including explicit Torch CPU factor, CUDA, historical scorer',
        'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('*.py')}}
    (ROOT/'checks/integration_tests_v21_running.json').write_text(json.dumps(value,indent=2)+'\n')
    try:
        import numpy,scipy,torch
        value['software']={'numpy':numpy.__version__,'scipy':scipy.__version__,'torch':torch.__version__}
        value['gpu']={'name':torch.cuda.get_device_name(0),'CUDA_VISIBLE_DEVICES':'1'}
        suite=unittest.defaultTestLoader.discover(str(SOURCE),pattern='test_*.py')
        with (ROOT/'checks/integration_tests_v21.log').open('x') as log,contextlib.redirect_stdout(log):
            result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        value.update(tests_run=result.testsRun,
            passed=result.testsRun-len(result.skipped)-len(result.failures)-len(result.errors),
            failed=len(result.failures),errors=len(result.errors),
            skipped=[{'test':str(t),'reason':why} for t,why in result.skipped])
        value['returncode']=0 if result.wasSuccessful() and not result.skipped and result.testsRun==64 else 1
        value['status']='all_current_checks_pass' if value['returncode']==0 else 'current_checks_incomplete_or_failed'
    except BaseException:
        value.update(status='test_execution_failed',returncode=1,error=traceback.format_exc())
    value['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipt.write_text(json.dumps(value,indent=2)+'\n')
    return value['returncode']


if __name__=='__main__':sys.exit(main())
