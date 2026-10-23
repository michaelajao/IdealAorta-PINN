"""Geometry-only full-box certification and explicit regular-grid vector axes.

The domain is the union of trilinear grid cells whose EIGHT mask vertices are
fluid. It is a conservative image-grid proxy, not a native CFD wall surface.
Neither certification nor sampling uses velocity values. The velocity-component
map is mandatory: image-array order must not silently become physical x/y/z.
"""
from dataclasses import dataclass,field
import numpy as np


def vector3(value,name):
    a=np.broadcast_to(np.asarray(value,float),(3,)).copy()
    if not np.isfinite(a).all():raise ValueError(f'finite {name} required')
    return a


@dataclass
class VoxelDomain:
    mask:np.ndarray
    spacing_m:np.ndarray
    origin_m:np.ndarray
    _prefix:np.ndarray=field(init=False,repr=False)
    _cell_shape:np.ndarray=field(init=False,repr=False)

    def __post_init__(self):
        self.mask=np.asarray(self.mask)
        if self.mask.dtype!=bool or self.mask.ndim!=3 or min(self.mask.shape)<2:
            raise ValueError('a boolean fluid-vertex mask with at least two grid vertices per axis is required')
        self.mask=self.mask.copy();self.mask.flags.writeable=False
        self.spacing_m=vector3(self.spacing_m,'grid spacing');self.origin_m=vector3(self.origin_m,'grid origin')
        if np.any(self.spacing_m<=0):raise ValueError('positive grid spacing required')
        shape=np.array(self.mask.shape)-1;cells=np.ones(shape,bool)
        for i in (0,1):
            for j in (0,1):
                for k in (0,1):cells&=self.mask[i:i+shape[0],j:j+shape[1],k:k+shape[2]]
        self._cell_shape=shape
        prefix=np.pad(cells.astype(np.int64),((1,0),(1,0),(1,0)))
        for axis in range(3):np.cumsum(prefix,axis=axis,out=prefix)
        self._prefix=prefix

    def certify_boxes(self,centres,width_m,margin_m=1e-4):
        x=np.asarray(centres,float);width=vector3(width_m,'box width')
        if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all() or np.any(width<=0):
            raise ValueError('finite (N,3) box centres and positive widths required')
        if not np.isfinite(margin_m) or margin_m<0:raise ValueError('nonnegative geometric margin required')
        half=width/2+margin_m
        lower=np.floor((x-half-self.origin_m)/self.spacing_m).astype(np.int64)
        upper=np.floor((x+half-self.origin_m)/self.spacing_m).astype(np.int64)
        valid=np.all((lower>=0)&(upper<self._cell_shape),axis=1)
        rows=np.flatnonzero(valid)
        if len(rows):
            lo=lower[rows];hi=upper[rows]+1;count=np.zeros(len(rows),np.int64)
            for i in (0,1):
                for j in (0,1):
                    for k in (0,1):
                        index=(hi[:,0] if i else lo[:,0],hi[:,1] if j else lo[:,1],hi[:,2] if k else lo[:,2])
                        count+=(-1)**(3-i-j-k)*self._prefix[index]
            valid[rows]=count==np.prod(hi-lo,axis=1)
        return valid

    def cartesian_centres(self,spacing_m,seed=0):
        spacing=float(spacing_m)
        if not np.isfinite(spacing) or spacing<=0:raise ValueError('positive observation spacing required')
        offset=np.random.default_rng(seed).uniform(0,spacing,3)
        high=self.origin_m+(np.array(self.mask.shape)-1)*self.spacing_m
        axes=[np.arange(self.origin_m[c]+offset[c],high[c],spacing) for c in range(3)]
        centres=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
        # Require the containing interpolation cell to be wholly fluid. Nearest
        # fluid-vertex membership alone can admit a point in a mixed cell.
        q=np.floor((centres-self.origin_m)/self.spacing_m).astype(int)
        valid=np.all((q>=0)&(q<self._cell_shape),axis=1)
        rows=np.flatnonzero(valid)
        for i in (0,1):
            for j in (0,1):
                for k in (0,1):valid[rows]&=self.mask[q[rows,0]+i,q[rows,1]+j,q[rows,2]+k]
        return centres[valid]


@dataclass
class LinearGridVelocity:
    values:np.ndarray   # (T,N0,N1,N2,3), stored components in their supplied order
    spacing_m:np.ndarray
    origin_m:np.ndarray
    component_to_grid_axes:np.ndarray

    def __post_init__(self):
        self.values=np.array(self.values,copy=True)
        if self.values.ndim!=5 or self.values.shape[-1]!=3 or min(self.values.shape[1:4])<2 or not np.isfinite(self.values).all():
            raise ValueError('finite regular-grid velocity (T,N0,N1,N2,3) required')
        self.values.flags.writeable=False
        self.spacing_m=vector3(self.spacing_m,'grid spacing');self.origin_m=vector3(self.origin_m,'grid origin')
        if np.any(self.spacing_m<=0):raise ValueError('positive grid spacing required')
        self.component_to_grid_axes=np.asarray(self.component_to_grid_axes,float).copy()
        m=self.component_to_grid_axes
        if m.shape!=(3,3) or not np.isfinite(m).all() or not np.array_equal(m@m.T,np.eye(3)) or not np.isin(m,[-1,0,1]).all():
            raise ValueError('an explicit signed permutation from stored components to grid axes is required')
        self.component_to_grid_axes.flags.writeable=False

    def predict(self,points,frame,chunk=16384):
        p=np.asarray(points,float)
        if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError('finite query points (N,3) required')
        if not 0<=frame<len(self.values):raise ValueError('frame outside supplied window')
        shape=np.array(self.values.shape[1:4]);output=[]
        for start in range(0,len(p),chunk):
            q=(p[start:start+chunk]-self.origin_m)/self.spacing_m
            tolerance=64*np.finfo(float).eps*shape.max()
            if np.any(q < -tolerance) or np.any(q > shape-1+tolerance):raise ValueError('query leaves the regular-grid interpolation support')
            q=np.clip(q,0,shape-1)  # Roundoff only, after the strict tolerance check.
            low=np.minimum(np.floor(q).astype(np.int64),shape-2);frac=q-low
            value=np.zeros((len(q),3))
            for i in (0,1):
                for j in (0,1):
                    for k in (0,1):
                        weight=(frac[:,0] if i else 1-frac[:,0])*(frac[:,1] if j else 1-frac[:,1])*(frac[:,2] if k else 1-frac[:,2])
                        value+=weight[:,None]*self.values[frame,low[:,0]+i,low[:,1]+j,low[:,2]+k]
            output.append(value@self.component_to_grid_axes.T)
        return np.concatenate(output) if output else np.empty((0,3))
