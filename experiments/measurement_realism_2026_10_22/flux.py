"""Area quadrature for the planar convex inlet/outlet caps of this geometry family.

The triangulation is reconstructed from exported coordinates. This is an explicit
surface approximation, not a claim to possess the original CFD face connectivity.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class PlanarCap:
    points: np.ndarray
    weights_m2: np.ndarray
    outward_normal: np.ndarray
    max_plane_error_m: float

    @classmethod
    def from_points(cls, points, interior_reference, relative_plane_tolerance=1e-6):
        from scipy.spatial import Delaunay
        p=np.asarray(points,dtype=float)
        reference=np.asarray(interior_reference,dtype=float)
        if p.ndim!=2 or p.shape[1]!=3 or len(p)<3 or reference.shape!=(3,):
            raise ValueError('finite cap points (N,3) and an interior reference are required')
        if not np.isfinite(p).all() or not np.isfinite(reference).all():
            raise ValueError('nonfinite cap geometry')
        _,first=np.unique(np.round(p,10),axis=0,return_index=True)
        p=p[np.sort(first)]
        centre=p.mean(0)
        _,singular,axes=np.linalg.svd(p-centre,full_matrices=False)
        if singular[1]<=1e-12:
            raise ValueError('cap must span a plane')
        normal=axes[2].copy()
        extent=float(np.ptp(p,axis=0).max())
        plane_error=float(np.abs((p-centre)@normal).max())
        if plane_error>max(1e-10,relative_plane_tolerance*extent):
            raise ValueError('cap is not planar within the declared tolerance')
        orientation=float(np.dot(reference-centre,normal))
        if abs(orientation)<1e-10:
            raise ValueError('interior reference does not identify an outward cap normal')
        if orientation>0:normal=-normal
        coordinates=(p-centre)@axes[:2].T
        triangles=Delaunay(coordinates).simplices
        a=coordinates[triangles[:,1]]-coordinates[triangles[:,0]]
        b=coordinates[triangles[:,2]]-coordinates[triangles[:,0]]
        area=.5*np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])
        weights=np.bincount(triangles.reshape(-1),weights=np.repeat(area/3,3),minlength=len(p))
        return cls(p.copy(),weights,normal,plane_error)

    def integrate(self,velocity):
        u=np.asarray(velocity,dtype=float)
        if u.shape!=self.points.shape or not np.isfinite(u).all():
            raise ValueError('finite cap velocity at every quadrature point is required')
        return float(self.weights_m2@(u@self.outward_normal))

    def metadata(self):
        return {'area_m2':float(self.weights_m2.sum()),'outward_normal':self.outward_normal.tolist(),
                'n_points':len(self.points),'max_plane_error_m':self.max_plane_error_m,
                'quadrature':'P1 integration on the planar convex-hull triangulation of exported cap points',
                'original_CFD_faces':False}


def cap_balance(predict,inlet,outlet):
    qi=inlet.integrate(predict(inlet.points))
    qo=outlet.integrate(predict(outlet.points))
    scale=.5*(abs(qi)+abs(qo))
    return {'inlet_outward_flux_m3_s':qi,'outlet_outward_flux_m3_s':qo,
            'net_cap_flux_m3_s':qi+qo,
            'relative_cap_imbalance':abs(qi+qo)/scale if scale>1e-15 else None,
            'scope':'cap imbalance only; full closed-boundary flux also requires wall flux',
            'inlet_geometry':inlet.metadata(),'outlet_geometry':outlet.metadata()}
