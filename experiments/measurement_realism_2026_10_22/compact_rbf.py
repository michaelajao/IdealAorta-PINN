"""Full-centre sparse C4-Wendland scalar and globally solenoidal comparators.

For q=(x-c)/ell and r=|q|, the vector kernel normalized to K(0)=I is
  (1-r)_+^4 [(1+4r-20r^2) I + 15 q q^T].
It is (3 ell^2/112)(Hessian-Laplacian I) applied to
  phi(r)=(1-r)_+^6(35r^2+18r+3)/3.
Thus every column is solenoidal, including across the support boundary. Sparse
support is a property of each global basis function, not local field switching.
Production regression also fits a low-order background (trace-free affine for
the vector model) from the same observations and boundary data. The sparse
kernel need not represent the entire large-scale through-flow on its own.

Construction: Lowitzsch (2002), Chapter III, equations 3.5 and Lemma 3.5:
https://oaktrust.library.tamu.edu/bitstreams/7f87c44d-9bf2-47e0-81ec-7bb7fc3b8475/download
Only the relevant construction was consulted; no full-thesis reading is claimed.
    """
from dataclasses import dataclass,field
import time
import numpy as np


class CompactFitError(RuntimeError):
    def __init__(self,trials):
        self.trials=trials
        super().__init__('No converged, boundary-admissible compact RBF candidate')


def solve_regularized(matrix,rhs,ridge,tol,max_iter,column_scaling=False,product_operator=None):
    """LSMR with optional right scaling of the SAME regularized objective.

    The scaled problem is [A D; sqrt(ridge) D] z = [b; 0], x=D z.
    Scaling A alone and retaining scalar damping would change the regularizer.
    The audit also retains the gradient residual in the original coordinates.
    """
    from scipy.sparse.linalg import lsmr,LinearOperator
    product=matrix if product_operator is None else product_operator
    if not column_scaling:
        s=lsmr(product,rhs,damp=np.sqrt(ridge),atol=tol,btol=tol,maxiter=max_iter)
        return s[0],{'istop':int(s[1]),'iterations':int(s[2]),'normr':float(s[3]),
                     'normal_residual':float(s[4]),'condition_estimate':float(s[6]),
                     'column_scaling':False}
    m,n=matrix.shape;root=np.sqrt(ridge)
    squared=np.asarray(matrix.power(2).sum(axis=0)).ravel()
    diagonal=1/np.sqrt(np.maximum(squared+ridge,np.finfo(float).tiny))
    def mv(z):
        x=diagonal*z
        return np.concatenate([product@x,root*x])
    def rmv(v):return diagonal*(product.T@v[:m]+root*v[m:])
    operator=LinearOperator((m+n,n),matvec=mv,rmatvec=rmv,dtype=float)
    augmented_rhs=np.concatenate([rhs,np.zeros(n)])
    remaining=int(max_iter);iterations=0;working_tol=float(tol);z=None;rounds=[]
    while remaining>0:
        s=lsmr(operator,augmented_rhs,atol=working_tol,btol=working_tol,maxiter=remaining,x0=z)
        z=s[0];used=int(s[2]);iterations+=used;remaining-=used
        x=diagonal*z;residual=product@x-rhs
        gradient=product.T@residual+ridge*x
        residual_norm=float(np.sqrt(np.dot(residual,residual)+ridge*np.dot(x,x)))
        gradient_norm=float(np.linalg.norm(gradient))
        # This is a Frobenius-normalized original-coordinate stationarity
        # check, distinct from LSMR's scaled stopping criterion.
        normalizer=float(np.sqrt(squared.sum()+ridge*n)*residual_norm)
        stationarity=gradient_norm/max(normalizer,np.finfo(float).tiny)
        rounds.append({'istop':int(s[1]),'iterations':used,'working_tolerance':working_tol,
                       'original_stationarity_Frobenius_normalized':stationarity})
        if stationarity<=10*tol or int(s[1]) not in (0,1,2,4,5) or not used:break
        tighter=max(np.finfo(float).eps,working_tol/10)
        if tighter==working_tol or len(rounds)>=8:break
        working_tol=tighter
    return x,{'istop':int(s[1]),'iterations':iterations,'normr':residual_norm,
              'normal_residual':gradient_norm,'condition_estimate':float(s[6]),
              'condition_estimate_coordinates':'right-scaled augmented system',
              'column_scaling':True,'scale_min':float(diagonal.min()),'scale_max':float(diagonal.max()),
              'original_stationarity_Frobenius_normalized':stationarity,
              'original_stationarity_tolerance':10*tol,'iteration_limit':int(max_iter),
              'refinement_rounds':rounds,'scaled_normal_residual':float(s[4])}


def pair_kernel(displacements,radius_m,vector=True):
    delta=np.asarray(displacements,float)
    radius=float(radius_m)
    if delta.ndim!=2 or delta.shape[1]!=3 or not np.isfinite(delta).all() or not np.isfinite(radius) or radius<=0:
        raise ValueError('finite displacements (N,3) and positive support radius required')
    q=delta/radius
    r=np.linalg.norm(q,axis=1)
    h=np.maximum(1-r,0.)
    if vector:
        return h[:,None,None]**4*((1+4*r-20*r*r)[:,None,None]*np.eye(3)+15*q[:,:,None]*q[:,None,:])
    return h**6*(35*r*r+18*r+3)/3


@dataclass
class CompactRBF:
    centres:np.ndarray
    radius_m:float
    vector:bool
    coefficients:np.ndarray|None=None
    polynomial:bool=True
    _tree:object=field(default=None,init=False,repr=False)
    _origin:object=field(default=None,init=False,repr=False)
    _length:float=field(default=1.,init=False,repr=False)

    def __post_init__(self):
        self.centres=np.array(self.centres,dtype=float,copy=True)
        if self.centres.ndim!=2 or self.centres.shape[1]!=3 or not len(self.centres) or not np.isfinite(self.centres).all():
            raise ValueError('finite nonempty basis centres (N,3) required')
        if not np.isfinite(self.radius_m) or self.radius_m<=0:
            raise ValueError('positive finite compact support radius required')
        self._origin=self.centres.mean(0)
        self._length=max(float(np.ptp(self.centres,axis=0).max()),1e-8)
        self.centres.flags.writeable=False

    def design(self,support,weights,chunk=64,max_nnz=120_000_000):
        from scipy.spatial import cKDTree
        from scipy.sparse import coo_matrix,vstack,hstack,csr_matrix
        points=np.asarray(support,float);w=np.asarray(weights,float)
        if points.ndim!=3 or points.shape[2]!=3 or not np.isfinite(points).all():
            raise ValueError('finite observation support (N,Q,3) required')
        if w.shape!=(points.shape[1],) or np.any(w<0) or not np.isclose(w.sum(),1):
            raise ValueError('nonnegative normalized quadrature weights required')
        if self._tree is None:self._tree=cKDTree(self.centres)
        blocks=[];total=0;factor=3 if self.vector else 1
        for start in range(0,len(points),chunk):
            p=points[start:start+chunk];row_parts=[];col_parts=[];value_parts=[]
            for q,weight in enumerate(w):
                neighbours=self._tree.query_ball_point(p[:,q],self.radius_m)
                count=np.fromiter((len(n) for n in neighbours),dtype=np.int64,count=len(p))
                if not count.sum():continue
                row=np.repeat(np.arange(len(p)),count)
                col=np.concatenate([np.asarray(n,dtype=np.int64) for n in neighbours if len(n)])
                k=pair_kernel(p[row,q]-self.centres[col],self.radius_m,self.vector)*weight
                if self.vector:
                    row_parts.append(np.broadcast_to(row[:,None,None]*3+np.arange(3)[None,:,None],k.shape).reshape(-1))
                    col_parts.append(np.broadcast_to(col[:,None,None]*3+np.arange(3)[None,None,:],k.shape).reshape(-1))
                    value_parts.append(k.reshape(-1))
                else:
                    row_parts.append(row);col_parts.append(col);value_parts.append(k)
            shape=(factor*len(p),factor*len(self.centres))
            if value_parts:
                matrix=coo_matrix((np.concatenate(value_parts),(np.concatenate(row_parts),np.concatenate(col_parts))),shape=shape).tocsr()
                matrix.eliminate_zeros()
            else:matrix=coo_matrix(shape).tocsr()
            if self.polynomial:
                mean=(np.einsum('nqc,q->nc',p,w)-self._origin)/self._length
                if self.vector:
                    affine=np.zeros((len(p),3,11));affine[:,:,:3]=np.eye(3)
                    col=3
                    for i in range(3):
                        for j in range(3):
                            if i!=j:affine[:,i,col]=mean[:,j];col+=1
                    affine[:,0,9],affine[:,2,9]=mean[:,0],-mean[:,2]
                    affine[:,1,10],affine[:,2,10]=mean[:,1],-mean[:,2]
                    affine=affine.reshape(3*len(p),11)
                else:affine=np.column_stack([np.ones(len(p)),mean])
                matrix=hstack([matrix,csr_matrix(affine)],format='csr')
            total+=matrix.nnz
            if total>max_nnz:raise MemoryError('compact RBF sparse matrix exceeds the declared nonzero budget')
            blocks.append(matrix)
        return vstack(blocks,format='csr')

    def observe(self,support,weights,chunk=256):
        if self.coefficients is None:raise ValueError('model not fitted')
        values=[]
        for start in range(0,len(support),chunk):
            value=self.design(support[start:start+chunk],weights)@self.coefficients
            values.append(np.asarray(value).reshape(-1,3))
        return np.concatenate(values)

    def predict(self,points,chunk=256):
        return self.observe(np.asarray(points)[:,None],np.ones(1),chunk)


def separated_centres(points,min_separation_m):
    """Deterministic geometry-only thinning; it never removes fitting rows.

    Near-coincident native wall points can otherwise give an unnecessarily
    ill-conditioned basis. Every supplied boundary condition still contributes
    to regression. Zero separation retains the original full-centre basis.
    """
    x=np.asarray(points,float)
    if not np.isfinite(min_separation_m) or min_separation_m<0:
        raise ValueError('nonnegative finite centre separation required')
    if min_separation_m==0:return x.copy()
    from scipy.spatial import cKDTree
    tree=cKDTree(x);available=np.ones(len(x),bool);chosen=[]
    for i in range(len(x)):
        if available[i]:
            chosen.append(i)
            available[tree.query_ball_point(x[i],min_separation_m)]=False
    return x[chosen].copy()


def full_centres(obs,wall_xyz,inlet_xyz,min_separation_m=0):
    centres=np.vstack([np.einsum('nqc,q->nc',obs.support_m[~obs.validation],obs.weights),wall_xyz,inlet_xyz])
    _,first=np.unique(np.round(centres,10),axis=0,return_index=True)
    return separated_centres(centres[np.sort(first)],min_separation_m)


def fit_compact_pair(obs,wall_xyz,inlet_xyz,inlet_values,vector,support_radii,
                     ridges,max_iter=3000,tol=1e-7,max_nnz=120_000_000,polynomial=True,
                     centre_separation_m=0,wall_weight=1.,boundary_tolerance=None,column_scaling=False,solver_device='cpu',
                     measurements_only=False,inlet_weight=2.,solver_method='lsmr',direct_factor_device='cpu',
                     direct_block_columns=512,direct_max_gpu_gib=42.,direct_max_host_gib=96.,direct_cpu_factor_backend='scipy'):
    from scipy.sparse import vstack
    from sparse_backend import sparse_products
    if measurements_only:
        if any(len(a) for a in (wall_xyz,inlet_xyz)) or np.asarray(inlet_values).size or boundary_tolerance is not None:
            raise ValueError('measurements-only fitting must not receive boundary targets or a boundary gate')
    elif not len(wall_xyz) or not len(inlet_xyz):
        raise ValueError('native fitting requires explicit known wall and inlet information')
    if solver_method not in ('lsmr','direct'):raise ValueError('unknown compact solver method')
    centres=full_centres(obs,wall_xyz,inlet_xyz,centre_separation_m)
    if not np.isfinite(wall_weight) or wall_weight<=0:raise ValueError('positive finite wall weight required')
    if not np.isfinite(inlet_weight) or inlet_weight<=0:raise ValueError('positive finite inlet weight required')
    if boundary_tolerance is not None and (not np.isfinite(boundary_tolerance) or boundary_tolerance<=0):
        raise ValueError('positive finite boundary tolerance required')
    mask=~obs.validation;val=obs.validation
    scale=max(float(np.quantile(np.linalg.norm(obs.values[:,mask],axis=2),.995)),1e-6)
    supports=[obs.support_m[mask]];weights=[obs.weights];targets_by_group=[obs.values[:,mask]]
    factors=[1/np.sqrt(mask.sum())]
    if not measurements_only:
        supports.extend([wall_xyz[:,None],inlet_xyz[:,None]]);weights.extend([np.ones(1),np.ones(1)])
        targets_by_group.extend([np.zeros((2,len(wall_xyz),3)),inlet_values])
        factors.extend([np.sqrt(wall_weight/len(wall_xyz)),np.sqrt(inlet_weight/len(inlet_xyz))])
    trials=[];best=np.inf;winner=None;selected=None
    for radius in support_radii:
        template=CompactRBF(centres,float(radius),vector,polynomial=polynomial)
        started=time.monotonic()
        try:
            pieces=[template.design(p,w,max_nnz=max_nnz) for p,w in zip(supports,weights)]
            matrix=vstack([m*f for m,f in zip(pieces,factors)],format='csr');del pieces
            if matrix.nnz>max_nnz:raise MemoryError('combined compact fit matrix exceeds its nonzero budget')
        except (MemoryError,ValueError) as exc:
            trial={'radius_m':float(radius),'ridge':None,'converged':False,'status':'assembly_failed',
                   'error':str(exc),'seconds':time.monotonic()-started}
            trials.append(trial);print({'event':'compact_trial',**trial},flush=True)
            continue
        assembly_s=time.monotonic()-started
        matrix_bytes=matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes
        print({'event':'compact_matrix','vector':vector,'radius_m':radius,'shape':matrix.shape,
               'nnz':matrix.nnz,'bytes':matrix_bytes,'assembly_s':assembly_s},flush=True)
        direct=None
        if solver_method=='direct':
            from direct_ridge import DirectRidgeSystem,GiB
            all_targets=[]
            for frame in range(2):
                targets=[v[frame] for v in targets_by_group]
                if vector:targets=[t.reshape(-1) for t in targets]
                all_targets.append(np.concatenate([v*f/scale for v,f in zip(targets,factors)]))
            rhs=np.stack(all_targets,axis=1) if vector else np.concatenate(all_targets,axis=1)
            try:
                direct=DirectRidgeSystem(matrix,rhs,device=solver_device,factor_device=direct_factor_device,
                    block_columns=direct_block_columns,max_gpu_bytes=int(direct_max_gpu_gib*GiB),
                    max_host_bytes=int(direct_max_host_gib*GiB),cpu_factor_backend=direct_cpu_factor_backend)
            except (MemoryError,RuntimeError,ValueError,ArithmeticError) as exc:
                trial={'radius_m':float(radius),'ridge':None,'converged':False,'status':'direct_setup_failed',
                       'error':str(exc),'seconds':time.monotonic()-started}
                trials.append(trial);print({'event':'compact_trial',**trial},flush=True)
                continue
            backend={'device':solver_device,'factor_device':direct_factor_device,
                     'cpu_factor_backend':direct_cpu_factor_backend,
                     'dtype':'float64','products':'column-blocked direct Gram; original sparse objective audit',
                     'memory_plan':direct.plan,'setup_seconds':direct.setup_seconds}
        else:product,backend=sparse_products(matrix,solver_device)
        for ridge in ridges:
            started=time.monotonic();models=[];stops=[];error=0.;boundary_errors=[];direct_record=None
            if direct is not None:
                try:
                    joint,direct_record=direct.solve(ridge,stationarity_tolerance=10*tol)
                except (MemoryError,RuntimeError,ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
                    trial={'radius_m':float(radius),'ridge':float(ridge),'converged':False,
                           'status':'direct_solve_failed','error':str(exc),'product_backend':backend,
                           'seconds':time.monotonic()-started}
                    trials.append(trial);print({'event':'compact_trial',**trial},flush=True)
                    continue
                if not direct_record['converged']:
                    trial={'radius_m':float(radius),'ridge':float(ridge),'converged':False,
                           'status':'original_stationarity_failed','direct_solver':direct_record,
                           'product_backend':backend,'seconds':time.monotonic()-started}
                    trials.append(trial);print({'event':'compact_trial',**trial},flush=True)
                    continue
            for frame in range(2):
                targets=[v[frame] for v in targets_by_group]
                if vector:targets=[t.reshape(-1) for t in targets]
                y=np.concatenate([v*f/scale for v,f in zip(targets,factors)])
                rhs=[y] if vector else [y[:,i] for i in range(3)]
                if direct is not None:
                    coef=joint[:,frame] if vector else joint[:,3*frame:3*frame+3]
                else:
                    solutions=[solve_regularized(matrix,b,ridge,tol,max_iter,column_scaling,product) for b in rhs]
                    coef=solutions[0][0] if vector else np.stack([s[0] for s in solutions],axis=1)
                model=CompactRBF(centres,float(radius),vector,coef*scale,polynomial=polynomial)
                predicted=model.observe(obs.support_m[val],obs.weights)
                error+=float(np.sum((predicted-obs.values[frame,val])**2))
                models.append(model)
                if boundary_tolerance is not None:
                    boundary_errors.extend([
                        float(np.sqrt(np.mean(np.sum(model.predict(wall_xyz)**2,axis=1))))/scale,
                        float(np.sqrt(np.mean(np.sum((model.predict(inlet_xyz)-inlet_values[frame])**2,axis=1))))/scale])
                if direct is None:stops.append([s[1] for s in solutions])
                print({'event':'compact_frame_solved','vector':vector,'radius_m':radius,'ridge':ridge,
                       'frame':frame,'solver_method':solver_method,'seconds_since_candidate_start':time.monotonic()-started},flush=True)
            error=float(np.sqrt(error/np.sum(obs.values[:,val]**2)))
            if direct is not None:converged=direct_record['converged']
            else:
                converged=all(s['istop'] in (0,1,2,4,5) for pair in stops for s in pair)
                if column_scaling:
                    converged=converged and all(s['original_stationarity_Frobenius_normalized']<=10*tol
                                                for pair in stops for s in pair)
            boundary_pass=boundary_tolerance is None or max(boundary_errors)<=boundary_tolerance
            trial={'radius_m':float(radius),'ridge':float(ridge),'val_observed_rel_l2':error,
                   'converged':converged,'lsmr':stops if direct is None else None,
                   'direct_solver':direct_record,'solver_method':solver_method,'seconds':time.monotonic()-started,
                   'matrix_shape':list(matrix.shape),'matrix_nnz':int(matrix.nnz),
                   'matrix_bytes':int(matrix_bytes),'assembly_seconds':assembly_s,
                   'boundary_rms_over_U':boundary_errors,'boundary_gate_pass':boundary_pass,'product_backend':backend}
            trials.append(trial);print({'event':'compact_trial',**trial},flush=True)
            if converged and boundary_pass and np.isfinite(error) and error<best:
                best=error;winner=models;selected={'radius_m':float(radius),'ridge':float(ridge)}
        if direct is not None:direct.close()
    if winner is None:raise CompactFitError(trials)
    return winner,{'basis':'global compact C4-Wendland; observation-fit and boundary centres',
                   'centre_min_separation_m':centre_separation_m,
                   'wall_weight':wall_weight,'boundary_tolerance':boundary_tolerance,
                   'inlet_weight':inlet_weight,
                   'column_scaling':bool(column_scaling and solver_method=='lsmr'),
                   'solver_device':solver_device,'solver_method':solver_method,
                   'direct_factor_device':direct_factor_device if solver_method=='direct' else None,
                   'direct_cpu_factor_backend':direct_cpu_factor_backend if solver_method=='direct' else None,
                   'boundary_contract':'measurements only; no wall/inlet/outlet targets' if measurements_only else 'known wall and inlet',
                   'selected_candidate':selected,
                   'n_fitting_wall_nodes':len(wall_xyz),'n_fitting_inlet_nodes':len(inlet_xyz),
                   'polynomial_background':'constant and trace-free affine' if vector and polynomial else ('affine' if polynomial else 'none'),
                   'basis_centres':len(centres),'global_solenoidal_basis':bool(vector),
                   'best_val_observed_rel_l2':best,'trials':trials,
                   'selection_uses_hidden_targets':False,'fit_policy':obs.metadata['fitting_policy']}
