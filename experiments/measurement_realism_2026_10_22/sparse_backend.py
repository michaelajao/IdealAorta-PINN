"""Optional float64 GPU products for SciPy's unchanged LSMR iteration.

Only A@x and A.T@y move to CUDA. SciPy retains its iteration, regularizer and
stopping tests. CPU/GPU roundoff can differ; equivalence must be checked and the
backend recorded. No mixed precision or normal-equation replacement is used.
"""
import time
import numpy as np


def sparse_products(matrix,device='cpu'):
    if device=='cpu':return matrix,{'device':'cpu','dtype':'float64','products':'SciPy CSR'}
    if not device.startswith('cuda'):raise ValueError('CPU or explicit CUDA device required')
    import torch
    from scipy.sparse.linalg import LinearOperator
    if not torch.cuda.is_available():raise RuntimeError('requested CUDA sparse products are unavailable')
    started=time.monotonic()
    def tensor(a):return torch.tensor(a,device=device)
    def convert(a):
        a=a.tocsr()
        return torch.sparse_csr_tensor(tensor(a.indptr),tensor(a.indices),tensor(a.data.astype(np.float64,copy=False)),
                                       size=a.shape,dtype=torch.float64,device=device)
    forward=convert(matrix);transpose=convert(matrix.T)
    def mv(v):return torch.mv(forward,torch.tensor(np.asarray(v).ravel(),dtype=torch.float64,device=device)).cpu().numpy()
    def rmv(v):return torch.mv(transpose,torch.tensor(np.asarray(v).ravel(),dtype=torch.float64,device=device)).cpu().numpy()
    operator=LinearOperator(matrix.shape,matvec=mv,rmatvec=rmv,dtype=np.float64)
    return operator,{'device':device,'dtype':'float64','products':'PyTorch CUDA CSR; SciPy LSMR iteration',
                     'setup_seconds':time.monotonic()-started,'gpu_name':torch.cuda.get_device_name(device),
                     'gpu_bytes_allocated_after_setup':int(torch.cuda.memory_allocated(device))}
