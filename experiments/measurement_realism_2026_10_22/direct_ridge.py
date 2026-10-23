"""Memory-bounded direct ridge solves for a sparse observation design.

The objective remains ||A X-B||_F^2 + ridge ||X||_F^2. A column-blocked
Gram product avoids materializing the full dense design. Normal equations can
lose accuracy, so acceptance uses stationarity recomputed with the ORIGINAL
sparse A and B, with optional factor-based iterative refinement. Neither a
successful Cholesky factorization nor its residual alone certifies a solve.
No dense work array is written to disk. CPU and CUDA allocation plans are
checked before CSC/Gram construction. CUDA Gram products can be combined with
CPU factorization when three dense Gram arrays would exceed GPU memory.
"""
import time
import numpy as np


GiB=1024**3


def add_direct_arguments(parser):
    parser.add_argument('--direct-factor-device',default='cpu',help='CPU or the explicit Gram CUDA device')
    parser.add_argument('--direct-cpu-factor-backend',choices=['scipy','torch'],default='scipy',
                        help='Explicit CPU factorization library; retained in execution contracts')
    parser.add_argument('--direct-block-columns',type=int,default=512)
    parser.add_argument('--direct-max-gpu-gib',type=float,default=42.)
    parser.add_argument('--direct-max-host-gib',type=float,default=96.)


def direct_settings(args):
    return {name:getattr(args,name) for name in ('direct_factor_device','direct_cpu_factor_backend','direct_block_columns',
            'direct_max_gpu_gib','direct_max_host_gib')}


def allocation_plan(shape,nnz,rhs_columns,block_columns,gram_device,factor_device):
    m,n=map(int,shape)
    if min(m,n,int(rhs_columns),int(block_columns))<=0 or int(nnz)<0:
        raise ValueError('positive design dimensions, RHS count and block width required')
    if gram_device!='cpu' and not gram_device.startswith('cuda:'):
        raise ValueError('CPU or an explicit CUDA device required')
    if factor_device not in ('cpu',gram_device):
        raise ValueError('factorization must use CPU or the Gram CUDA device')
    block=min(n,int(block_columns));gram=8*n*n
    # Worst-case float64 data plus int64 sparse indices, including transposes.
    sparse=16*int(nnz)+8*(m+n+2)
    column_block=8*m*block;product_block=8*n*block
    rhs_work=8*(m+n)*int(rhs_columns)*6
    host=4*gram+3*sparse+2*column_block+2*product_block+rhs_work+64*1024**2
    gpu=0
    if gram_device!='cpu':
        gram_copies=3 if factor_device!='cpu' else 0
        gpu=gram_copies*gram+sparse+2*column_block+2*product_block+rhs_work+2*GiB
    return {'host_peak_bound_bytes':host,'gpu_peak_bound_bytes':gpu,
            'dense_gram_bytes':gram,'dense_design_bytes_avoided':8*m*n,
            'block_columns':block,'gram_device':gram_device,'factor_device':factor_device,
            'host_gram_streaming':gram_device!='cpu' and factor_device=='cpu',
            'dense_block_order':'C on CUDA; F on CPU',
            'assumption':'conservative live arrays plus workspace reserve; no disk spill'}


def _host_available():
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        import sys
        if sys.platform.startswith('linux'):
            with open('/proc/meminfo') as stream:
                for line in stream:
                    if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
        raise RuntimeError('available-host-memory information is required; install psutil on this platform')


class DirectRidgeSystem:
    def __init__(self,matrix,rhs,device='cpu',factor_device='cpu',block_columns=512,
                 max_gpu_bytes=42*GiB,max_host_bytes=96*GiB,cpu_factor_backend='scipy'):
        from scipy.sparse import isspmatrix_csr
        if not isspmatrix_csr(matrix) or matrix.dtype!=np.float64:
            raise ValueError('a float64 SciPy CSR design is required')
        b=np.asarray(rhs,dtype=np.float64)
        if b.ndim==1:b=b[:,None]
        if b.ndim!=2 or b.shape[0]!=matrix.shape[0] or not np.isfinite(b).all() or not np.isfinite(matrix.data).all():
            raise ValueError('finite design and matching finite right-hand sides required')
        if max_gpu_bytes<=0 or max_host_bytes<=0:
            raise ValueError('positive explicit memory bounds required')
        if cpu_factor_backend not in ('scipy','torch'):raise ValueError('unknown CPU factorization backend')
        self.cpu_factor_backend=cpu_factor_backend
        self.plan=allocation_plan(matrix.shape,matrix.nnz,b.shape[1],block_columns,device,factor_device)
        self.plan['cpu_factor_backend']=cpu_factor_backend
        available_host=_host_available()
        host_limit=min(int(max_host_bytes),max(0,available_host-512*1024**2))
        if self.plan['host_peak_bound_bytes']>host_limit:
            raise MemoryError('direct ridge host allocation bound exceeds available/declared memory')
        self.device=device;self.factor_device=factor_device;self.matrix=matrix;self.rhs=b
        self.frobenius_squared=float(np.dot(matrix.data,matrix.data))
        self.plan['available_host_bytes_before_setup']=available_host
        self.plan['effective_host_limit_bytes']=host_limit
        started=time.monotonic();m,n=matrix.shape;block=self.plan['block_columns']
        transpose_gpu=None
        if device=='cpu':
            gram=np.empty((n,n),dtype=np.float64,order='F')
        else:
            import torch
            if not torch.cuda.is_available():raise RuntimeError('requested CUDA Gram products are unavailable')
            torch.cuda.empty_cache()
            free,total=torch.cuda.mem_get_info(device)
            gpu_limit=min(int(max_gpu_bytes),max(0,int(free)-GiB))
            self.plan.update(available_gpu_bytes_before_setup=int(free),gpu_total_bytes=int(total),
                             effective_gpu_limit_bytes=gpu_limit)
            if self.plan['gpu_peak_bound_bytes']>gpu_limit:
                raise MemoryError('direct ridge GPU allocation bound exceeds available/declared memory')
            gram=np.empty((n,n),dtype=np.float64,order='F') if factor_device=='cpu' else torch.empty((n,n),dtype=torch.float64,device=device)
        columns=matrix.tocsc()
        if device!='cpu':
            transpose=columns.T.tocsr(copy=False)
            transpose_gpu=torch.sparse_csr_tensor(
                torch.tensor(transpose.indptr,device=device),torch.tensor(transpose.indices,device=device),
                torch.tensor(transpose.data,device=device),size=(n,m),dtype=torch.float64,device=device)
            del transpose
        for start in range(0,n,block):
            stop=min(start+block,n)
            # CSC defaults to column-major. Row-contiguous CUDA RHS blocks avoid
            # strided global-memory reads inside sparse matrix multiplication.
            dense=columns[:,start:stop].toarray(order='F' if device=='cpu' else 'C')
            if device=='cpu':gram[:,start:stop]=matrix.T@dense
            else:
                dense_gpu=torch.as_tensor(dense,dtype=torch.float64,device=device)
                product=torch.sparse.mm(transpose_gpu,dense_gpu)
                if factor_device=='cpu':gram[:,start:stop]=product.cpu().numpy()
                else:gram[:,start:stop]=product
                del product,dense_gpu
            del dense
        del columns,transpose_gpu
        self.normal_rhs=np.asarray(matrix.T@b)
        if device!='cpu':
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self.gram=gram
        # A sampled symmetry audit avoids allocating a second full Gram matrix.
        ids=np.linspace(0,n-1,min(n,64),dtype=int)
        if factor_device=='cpu':sample=self.gram[np.ix_(ids,ids)]
        else:sample=self.gram[torch.tensor(ids,device=device)[:,None],torch.tensor(ids,device=device)[None,:]].cpu().numpy()
        if not np.isfinite(sample).all() or not np.allclose(sample,sample.T,rtol=1e-10,atol=1e-12):
            raise ArithmeticError('formed Gram failed the finite/symmetry sample check')
        self.setup_seconds=time.monotonic()-started

    def solve(self,ridge,stationarity_tolerance=1e-6,max_refinement=3):
        if not np.isfinite(ridge) or ridge<0 or not np.isfinite(stationarity_tolerance) or stationarity_tolerance<=0:
            raise ValueError('finite nonnegative ridge and positive stationarity tolerance required')
        if not isinstance(max_refinement,int) or max_refinement<0:
            raise ValueError('nonnegative integer refinement budget required')
        if self.gram is None:raise ValueError('direct ridge system is closed')
        started=time.monotonic();n=self.matrix.shape[1]
        if self.factor_device=='cpu' and self.cpu_factor_backend=='scipy':
            from scipy.linalg import cho_factor,cho_solve
            regularized=np.array(self.gram,dtype=np.float64,order='F',copy=True)
            regularized.flat[::n+1]+=ridge
            factor=cho_factor(regularized,lower=True,overwrite_a=True,check_finite=True)
            del regularized
            def solve_factor(y):return cho_solve(factor,y,check_finite=True)
            factor_backend='SciPy CPU Cholesky'
        else:
            try:import torch
            except ImportError as exc:raise RuntimeError('requested Torch factorization requires PyTorch') from exc
            regularized=(torch.from_numpy(self.gram).clone() if self.factor_device=='cpu' else self.gram.clone())
            regularized.diagonal().add_(ridge)
            factor,info=torch.linalg.cholesky_ex(regularized,check_errors=False)
            del regularized
            code=int(info.cpu())
            if code:raise np.linalg.LinAlgError('Torch Cholesky factorization failed at minor '+str(code))
            def solve_factor(y):
                return torch.cholesky_solve(torch.as_tensor(y,dtype=torch.float64,device=self.factor_device),factor).cpu().numpy()
            factor_backend='Torch '+self.factor_device+' Cholesky'
        x=solve_factor(self.normal_rhs);rounds=[]
        for refinement in range(max_refinement+1):
            residual=self.matrix@x-self.rhs
            gradient=self.matrix.T@residual+ridge*x
            objective=np.sum(residual*residual,axis=0)+ridge*np.sum(x*x,axis=0)
            denominator=np.sqrt((self.frobenius_squared+ridge*n)*objective)
            stationarity=np.linalg.norm(gradient,axis=0)/np.maximum(denominator,np.finfo(float).tiny)
            relative_normal=np.linalg.norm(gradient,axis=0)/np.maximum(
                np.linalg.norm(self.normal_rhs,axis=0)+ridge*np.linalg.norm(x,axis=0),np.finfo(float).tiny)
            rounds.append({'refinement':refinement,'original_stationarity_Frobenius_normalized':stationarity.tolist(),
                           'original_gradient_over_normal_rhs':relative_normal.tolist()})
            finite=bool(np.isfinite(x).all() and np.isfinite(objective).all() and np.isfinite(stationarity).all())
            if not finite:raise ArithmeticError('direct ridge produced nonfinite original-system values')
            converged=finite and bool(np.all(stationarity<=stationarity_tolerance))
            if converged or not finite or refinement==max_refinement:break
            x+=solve_factor(-gradient)
        record={'method':'column-blocked direct ridge; original-objective audit','converged':converged,
                'factor_backend':factor_backend,
                'ridge':float(ridge),'original_stationarity_tolerance':stationarity_tolerance,
                'original_stationarity_Frobenius_normalized':stationarity.tolist(),
                'original_gradient_over_normal_rhs':relative_normal.tolist(),
                'regularized_objective_per_rhs':objective.tolist(),'refinement_rounds':rounds,
                'seconds':time.monotonic()-started}
        return x,record

    def close(self):
        self.gram=None
        if self.device!='cpu':
            import torch
            torch.cuda.empty_cache()
