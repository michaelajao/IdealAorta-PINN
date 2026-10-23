"""Geometry-only certification against an explicitly declared point-cloud proxy.

The original CFD connectivity/CAD is unavailable. This proxy is NOT an exact
certificate of the native CFD lumen. Every observation receipt records that limit.
"""
from __future__ import annotations
import numpy as np


class OrientedPointDomain:
    """Inside iff the nearest boundary point's inward tangent plane is positive.

    Full-box certification checks ALL boundary points that could be nearest to
    ANY point in the box, not just its corners or quadrature nodes. Its guarantee
    applies to this reconstructed domain, conditional on the supplied normals.
    """
    def __init__(self, boundary_xyz, inward_normals, margin_m=0.0001):
        from scipy.spatial import cKDTree
        self.xyz = np.asarray(boundary_xyz, dtype=float)
        self.normals = np.asarray(inward_normals, dtype=float)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3 or self.normals.shape != self.xyz.shape:
            raise ValueError("boundary and normals must be matching (N,3)")
        if not np.isfinite(self.xyz).all() or not np.isfinite(self.normals).all():
            raise ValueError("nonfinite boundary geometry")
        lengths = np.linalg.norm(self.normals, axis=1)
        if np.any(lengths < 1e-12) or margin_m < 0:
            raise ValueError("nonzero inward normals and nonnegative margin required")
        self.normals = self.normals / lengths[:, None]
        self.tree = cKDTree(self.xyz)
        self.margin = float(margin_m)

    def contains(self, points):
        p = np.asarray(points, dtype=float)
        _, j = self.tree.query(p)
        return np.einsum("ij,ij->i", p - self.xyz[j], self.normals[j]) >= self.margin

    def certify_boxes(self, centres, width_m):
        c = np.asarray(centres, dtype=float)
        half = np.broadcast_to(np.asarray(width_m, dtype=float), (3,)) / 2
        if np.any(half <= 0):
            raise ValueError("positive finite box widths required")
        radius = float(np.linalg.norm(half))
        nearest, _ = self.tree.query(c)
        result = np.zeros(len(c), bool)
        for i, centre in enumerate(c):
            # If w is nearest to y in the box and w0 nearest to centre:
            # |w-centre| <= |w-y|+|y-centre| <= |w0-centre|+2*radius.
            js = self.tree.query_ball_point(centre, np.nextafter(nearest[i] + 2 * radius, np.inf))
            n = self.normals[js]
            minimum = np.einsum("ij,ij->i", centre - self.xyz[js], n) - np.abs(n) @ half
            result[i] = np.all(minimum >= self.margin)
        return result

    def metadata(self):
        return {"domain": "nearest-oriented-boundary-point implicit proxy",
                "native_CFD_lumen_certified": False,
                "full_boxes_certified_in_proxy": True,
                "margin_m": self.margin,
                "limitation": "no original surface connectivity/CAD; normal and surface sampling uncertainty remain"}
