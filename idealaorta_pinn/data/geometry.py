"""Wall-surface normal estimation.

Estimates inward-pointing unit normals on the wall point cloud (from the WSS
export), used by the no-slip BC and the autodiff WSS loss. Uses Open3D when
available (radius-based neighborhoods, consistent orientation), otherwise a
local-PCA fallback. Mirrors the approach in the group's prior projects.
"""

from __future__ import annotations

import importlib.util

import numpy as np
from sklearn.neighbors import NearestNeighbors

_HAS_OPEN3D = importlib.util.find_spec("open3d") is not None


def estimate_normals_pca(points: np.ndarray, k: int = 16) -> np.ndarray:
    """Inward unit normals via local PCA over k nearest neighbors.

    The smallest-eigenvalue eigenvector of the local covariance is the surface
    normal; orientation is flipped toward the global centroid (inward).
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    k = max(3, min(k, n))
    idx = NearestNeighbors(n_neighbors=k).fit(pts).kneighbors(pts, return_distance=False)

    normals = np.zeros_like(pts)
    for i in range(n):
        nbrs = pts[idx[i]]
        cov = np.cov((nbrs - nbrs.mean(axis=0)).T)
        w, v = np.linalg.eigh(cov)
        nrm = v[:, 0]
        normals[i] = nrm / (np.linalg.norm(nrm) + 1e-12)

    centroid = pts.mean(axis=0, keepdims=True)
    to_center = centroid - pts
    flip = np.sign((normals * to_center).sum(axis=1, keepdims=True))
    normals = normals * np.where(flip >= 0, 1.0, -1.0)
    return normals


def estimate_normals_open3d(points: np.ndarray, radius_mult: float = 3.0,
                            max_nn: int = 64, orient_k: int = 50) -> np.ndarray:
    """Inward unit normals via Open3D (radius-hybrid + consistent orientation)."""
    if not _HAS_OPEN3D:
        return estimate_normals_pca(points)
    import open3d as o3d  # type: ignore

    pts = np.asarray(points, dtype=np.float64)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    med = float(np.median(dists)) if dists.size else 1e-3
    radius = max(radius_mult * med, 1e-6)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.orient_normals_consistent_tangent_plane(orient_k)
    nrm = np.asarray(pcd.normals)
    centroid = pts.mean(axis=0, keepdims=True)
    flip = np.sign((nrm * (centroid - pts)).sum(axis=1, keepdims=True))
    return nrm * np.where(flip >= 0, 1.0, -1.0)


def compute_wall_normals(points: np.ndarray, method: str = "auto", k: int = 16) -> np.ndarray:
    """Compute inward unit normals; ``method`` in {'auto','open3d','pca'}."""
    if method == "pca" or (method == "auto" and not _HAS_OPEN3D):
        return estimate_normals_pca(points, k=k)
    return estimate_normals_open3d(points)


# ---------------------------------------------------------------------------
# Inlet / outlet cross-section detection and sampling (standardized coords)
# ---------------------------------------------------------------------------
def detect_inlet_outlet(coords_std: np.ndarray, tol_frac: float = 0.02) -> dict:
    """Detect inlet/outlet cross-sections from standardized wall coordinates.

    The axial direction is the coordinate of largest span; inlet = axial min,
    outlet = axial max. Returns centers/radii in standardized coordinates.
    """
    coords = np.asarray(coords_std, dtype=np.float64)
    spans = coords.max(axis=0) - coords.min(axis=0)
    axial = int(np.argmax(spans))
    others = [d for d in range(3) if d != axial]
    tol = tol_frac * spans[axial]

    out: dict = {"axial_dim": axial}
    a = coords[:, axial]
    for label, fn, cmp in (("inlet", np.min, np.less), ("outlet", np.max, np.greater)):
        extreme = fn(a)
        mask = cmp(a, extreme + tol) if label == "inlet" else cmp(a, extreme - tol)
        ring = coords[mask]
        center = ring[:, others].mean(axis=0)
        radius = float(np.sqrt(((ring[:, others] - center) ** 2).sum(axis=1)).mean())
        out[f"{label}_axial_pos"] = float(ring[:, axial].mean())
        out[f"{label}_center"] = center.tolist()
        out[f"{label}_radius"] = radius
    return out


def cross_section_points(axial_pos: float, center, radius: float, axial_dim: int,
                         n_radial: int = 6, n_angular: int = 12) -> np.ndarray:
    """Disk of sample points at a cross-section (standardized coords), ``(M,3)``."""
    others = [d for d in range(3) if d != axial_dim]
    pts2d = [[0.0, 0.0]]
    for ir in range(1, n_radial + 1):
        r = radius * (ir / n_radial) * 0.95
        for ia in range(n_angular):
            th = 2.0 * np.pi * ia / n_angular
            pts2d.append([r * np.cos(th), r * np.sin(th)])
    pts2d = np.asarray(pts2d)
    coords = np.zeros((pts2d.shape[0], 3))
    coords[:, axial_dim] = axial_pos
    coords[:, others[0]] = center[0] + pts2d[:, 0]
    coords[:, others[1]] = center[1] + pts2d[:, 1]
    return coords
