"""Unchanged historical pressure integration with its solver diagnostics retained."""
import time
import numpy as np
from historical import api_module


def absolute_wss_scores(relative,reference,mask):
    """Recover RMS Pa from the historical node-weighted relative L2 scores.

    Both magnitude and vector relative errors use ||reference||_F. Multiplying
    by its vector RMS therefore gives their absolute RMS on exactly that mask,
    without repeating interpolation or changing the wall-shear estimator.
    """
    reference=np.asarray(reference,dtype=float);mask=np.asarray(mask,dtype=bool)
    if reference.ndim!=2 or reference.shape[1]!=3 or mask.shape!=(len(reference),) or not mask.any():
        raise ValueError('nonempty matching wall-shear mask and (N,3) reference required')
    values=np.array([relative['wss_mag_rel_l2'],relative['wss_vec_rel_l2']],dtype=float)
    if not np.isfinite(reference[mask]).all() or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('finite reference and nonnegative finite relative errors required')
    rms=float(np.sqrt(np.mean(np.sum(reference[mask]**2,axis=1))))
    if rms<=0:raise ValueError('positive wall-shear reference RMS required')
    return {'wss_mag_rms_Pa':float(values[0]*rms),'wss_vec_rms_Pa':float(values[1]*rms),
            'reference_vector_rms_Pa':rms,'n_scored_wall_nodes':int(mask.sum()),
            'absolute_error_scope':'same aneurysm-zone mask and node weighting as historical relative L2'}


def make_audited_integrator(P):
    base=api_module('postprocess').PressureIntegrator
    class AuditedIntegrator(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.solve_records=[]

        def integrate(self,f_int,iters=6000):
            from scipy.sparse.linalg import lsqr
            started=time.monotonic()
            rhs=np.r_[f_int[self.sub].reshape(-1),0.]
            solved=lsqr(self.A,rhs,atol=1e-10,btol=1e-10,iter_lim=iters)
            self.solve_records.append({'istop':int(solved[1]),'iterations':int(solved[2]),
                'residual_norm':float(solved[3]),'condition_estimate':float(solved[6]),
                'normal_residual':float(solved[7]),'converged':int(solved[1]) in (0,1,2,4,5),
                'atol':1e-10,'btol':1e-10,'iteration_limit':iters,'seconds':time.monotonic()-started})
            p=solved[0]
            return p-p[self.o_idx].mean()
    return AuditedIntegrator(P.X,P.wall_xyz,P.dwall)
