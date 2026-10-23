import json,time,pathlib,datetime,numpy as np,torch
root=pathlib.Path('/home/olarinoyem/measurement_realism_2026_10_22');device='cuda:0'
free,_=torch.cuda.mem_get_info(device)
if free<4*1024**3:raise MemoryError('layout probe needs its explicit small GPU reserve')
rng=np.random.default_rng(20261023);m=121392;n=2048;k=1024;width=512
indices=np.sort(np.stack([rng.choice(m,k,replace=False) for _ in range(n)]),axis=1)
values=rng.normal(size=n*k);pointers=np.arange(n+1,dtype=np.int64)*k
matrix=torch.sparse_csr_tensor(torch.tensor(pointers,device=device),torch.tensor(indices.ravel(),device=device),torch.tensor(values,device=device),size=(n,m),dtype=torch.float64,device=device)
a=rng.normal(size=(m,width));arrays={order:torch.as_tensor(np.array(a,order=order),device=device) for order in ('C','F')}
rows=[];reference=None
for order in ['C','F','F','C','C','F']:
 torch.cuda.synchronize();started=time.monotonic();out=torch.sparse.mm(matrix,arrays[order]);torch.cuda.synchronize();seconds=time.monotonic()-started
 current=out.cpu().numpy()
 if reference is None:reference=current.copy()
 np.testing.assert_allclose(current,reference,atol=2e-10,rtol=2e-10)
 row={'order':order,'seconds':seconds,'input_stride':list(arrays[order].stride()),'contiguous':arrays[order].is_contiguous()};rows.append(row);print(json.dumps(row),flush=True)
receipt={'scope':'manufactured sparse product and layout check on shared GPU; not CFD accuracy or an isolated hardware benchmark','shape':[n,m],'nnz':n*k,'RHS_columns':width,'trials':rows,'outputs_equivalent':True,'GPU':torch.cuda.get_device_name(0),'torch':torch.__version__,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(root/'checks/layout_probe_v20.json').write_text(json.dumps(receipt,indent=2)+'\n')
