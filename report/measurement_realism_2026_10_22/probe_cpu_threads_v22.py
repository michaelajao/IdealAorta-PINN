"""Bounded shared-machine CPU Cholesky timing; no CFD/evaluation data."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import time

ROOT=Path('/home/olarinoyem/measurement_realism_2026_10_22')


def cpu_times():
    row=Path('/proc/stat').read_text().splitlines()[0].split()[1:]
    values=list(map(int,row[:8]))
    return sum(values),values[3]+values[4]


def main():
    assert 'brosnan' in socket.gethostname().lower()
    output=ROOT/'checks/cpu_threads_probe_v22.json'
    assert not output.exists(),'retain previous benchmark'
    value={'status':'running','pid':os.getpid(),
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'manufactured 8192-square CPU factorization on a shared machine; not a native or universal speed claim',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'allowed_logical_CPUs':len(os.sched_getaffinity(0))}
    output.write_text(json.dumps(value,indent=2)+'\n')
    first=cpu_times();time.sleep(1);last=cpu_times()
    value['observed_aggregate_CPU_busy_fraction']=1-(last[1]-first[1])/(last[0]-first[0])
    import torch
    torch.set_num_interop_threads(1);torch.set_num_threads(4)
    n=8192;alpha=.01
    a=torch.empty_strided((n,n),(1,n),dtype=torch.float64)
    a.fill_(alpha);a.diagonal().add_(1.)
    generator=torch.Generator().manual_seed(73002)
    b=torch.randn(n,2,dtype=torch.float64,generator=generator)
    expected=b-alpha*b.sum(0,keepdim=True)/(1+alpha*n)
    # Warm the library separately from all recorded large solves.
    torch.linalg.cholesky_ex(torch.eye(256,dtype=torch.float64))
    rows=[]
    for threads in (4,8,4,8,4,8):
        torch.set_num_threads(threads)
        started=time.monotonic()
        factor,info=torch.linalg.cholesky_ex(a,check_errors=False)
        assert int(info)==0
        x=torch.cholesky_solve(b,factor)
        seconds=time.monotonic()-started
        error=float(torch.max(torch.abs(x-expected)))
        residual=float(torch.linalg.vector_norm(x+alpha*x.sum(0,keepdim=True)-b)/torch.linalg.vector_norm(b))
        assert error<1e-9 and residual<1e-10
        rows.append({'threads':threads,'factor_and_solve_seconds':seconds,
                     'max_error_against_analytic_solution':error,'relative_residual':residual})
        del factor,x
    import statistics
    medians={str(t):statistics.median(r['factor_and_solve_seconds'] for r in rows if r['threads']==t) for t in (4,8)}
    value.update(status='manufactured_thread_probe_complete',rows=rows,median_seconds=medians,
                 median_4_to_8_elapsed_ratio=medians['4']/medians['8'],
                 finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value),flush=True)


if __name__=='__main__':main()
