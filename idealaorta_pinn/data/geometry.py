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
_OPEN3D_WARNED = False


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
    try:
        import open3d as o3d  # type: ignore
    except Exception as e:  # noqa: BLE001
        # open3d is installed but its import chain can break (it pulls in
        # dash -> requests -> a half-compiled charset_normalizer). A findable
        # module is not necessarily importable, so fall back to PCA normals.
        # Warn only once -- this is called per (case, phase) and would spam logs.
        global _OPEN3D_WARNED
        if not _OPEN3D_WARNED:
            import warnings
            warnings.warn(f"open3d import failed ({type(e).__name__}); using PCA wall normals "
                          "(this message is shown once)", RuntimeWarning)
            _OPEN3D_WARNED = True
        return estimate_normals_pca(points)

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
    """Compute inward unit normals; ``method`` in {'auto','open3d','pca'}.

    ``auto`` uses Open3D when it is importable and falls back to PCA otherwise.
    Note ``k`` (PCA neighbourhood size) applies to the PCA path ONLY — the Open3D
    path uses its own radius/max-nn heuristics, so tuning ``k`` has no effect
    unless you also force ``method='pca'``.
    """
    if method == "pca" or (method == "auto" and not _HAS_OPEN3D):
        return estimate_normals_pca(points, k=k)
    return estimate_normals_open3d(points)


def _signed_lumen_distance(query: np.ndarray, tree, wall_coords: np.ndarray,
                           wall_normals: np.ndarray) -> np.ndarray:
    """Local signed distance into the lumen for each ``query`` point.

    Projects the vector from the nearest wall point onto that wall point's INWARD
    normal; positive = inside. ``tree`` is a prebuilt cKDTree over ``wall_coords``.
    """
    _, idx = tree.query(query)
    return np.einsum("ij,ij->i", query - wall_coords[idx], wall_normals[idx])


def lumen_inside_mask(query: np.ndarray, wall_coords: np.ndarray,
                      wall_normals: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Boolean mask: which ``query`` points lie inside the lumen wall.

    A point is interior when the vector from its nearest wall point to it has a
    POSITIVE projection on that wall point's INWARD normal (local signed distance
    > ``margin``). The nearest-wall + inward-normal test follows the boundary into
    the saccular bulge, so it does not leak across the non-convex geometry the way
    a convex hull would. ``query`` and ``wall_coords`` must share one coordinate
    frame (and ``wall_normals`` must be the inward normals for ``wall_coords``).
    """
    from scipy.spatial import cKDTree

    query = np.asarray(query, dtype=np.float64)
    wall_coords = np.asarray(wall_coords, dtype=np.float64)
    wall_normals = np.asarray(wall_normals, dtype=np.float64)
    tree = cKDTree(wall_coords)
    return _signed_lumen_distance(query, tree, wall_coords, wall_normals) > margin


def sample_lumen_interior(wall_coords: np.ndarray, wall_normals: np.ndarray,
                          n_points: int, rng: np.random.Generator,
                          margin: float = 0.0, oversample: int = 12,
                          max_batches: int = 200,
                          bbox: tuple | None = None) -> np.ndarray:
    """S2: uniform rejection-sample ``n_points`` inside the (non-convex) lumen.

    A candidate is interior when the vector from its nearest wall point to it has
    a POSITIVE projection on that wall point's INWARD normal (a local signed
    distance > ``margin``). The nearest-wall + inward-normal test follows the
    boundary into the saccular bulge, so it does not leak across the non-convex
    geometry the way a convex hull / tube would. Returns ``(<=n_points, 3)`` in
    the same (standardized) coordinates as ``wall_coords``. Pass ``bbox=(lo, hi)``
    to restrict sampling to a sub-region (e.g. the saccular bulge band).
    """
    from scipy.spatial import cKDTree

    wall_coords = np.asarray(wall_coords, dtype=np.float64)
    wall_normals = np.asarray(wall_normals, dtype=np.float64)
    tree = cKDTree(wall_coords)
    if bbox is None:
        lo, hi = wall_coords.min(axis=0), wall_coords.max(axis=0)
    else:
        lo, hi = np.asarray(bbox[0], float), np.asarray(bbox[1], float)

    kept: List[np.ndarray] = []
    have = 0
    for _ in range(max_batches):
        if have >= n_points:
            break
        batch = rng.uniform(lo, hi, size=(max((n_points - have) * oversample, 2048), 3))
        signed = _signed_lumen_distance(batch, tree, wall_coords, wall_normals)
        inside = batch[signed > margin]
        if len(inside):
            kept.append(inside)
            have += len(inside)
    if not kept:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(kept)[:n_points]


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


def sac_axial_band(wall_coords: np.ndarray, n_bins: int = 24,
                   end_frac: float = 0.15) -> dict:
    """Locate the saccular bulge as the interior axial band of maximum lumen radius.

    Bins the wall cloud along its largest-span (axial) axis, measures each bin's
    transverse radius (mean distance from the bin's transverse centroid), and takes
    the interior bin (excluding the end ``end_frac`` at each end -- inlet/outlet) of
    largest radius as the sac centre. The band is then grown contiguously around the
    peak down to half its prominence over the baseline (median) radius, giving an
    x-range that brackets the bulge. Returns a dict with ``axial_dim``, band
    ``x_lo``/``x_hi``/``x_center``/``half_width``, transverse ``center2d``, and peak
    ``radius`` (same frame as ``wall_coords``).

    Healthy (taper-only) geometries have no interior bulge, so the band is shallow
    and arbitrary; callers should gate sac-specific outputs to diseased cases.
    """
    coords = np.asarray(wall_coords, dtype=np.float64)
    spans = coords.max(axis=0) - coords.min(axis=0)
    axial = int(np.argmax(spans))
    others = [d for d in range(3) if d != axial]
    a = coords[:, axial]
    lo, hi = float(a.min()), float(a.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    radii = np.full(n_bins, -1.0)
    cen2d = np.zeros((n_bins, 2))
    for b in range(n_bins):
        upper = a <= edges[b + 1] if b == n_bins - 1 else a < edges[b + 1]
        m = (a >= edges[b]) & upper
        if int(m.sum()) < 3:
            continue
        ring = coords[m][:, others]
        c = ring.mean(axis=0)
        radii[b] = float(np.sqrt(((ring - c) ** 2).sum(axis=1)).mean())
        cen2d[b] = c

    valid = radii > 0
    span = hi - lo
    interior = valid & (centers >= lo + end_frac * span) & (centers <= hi - end_frac * span)
    pool = interior if interior.any() else valid
    idx = np.where(pool)[0]
    peak = int(idx[np.argmax(radii[idx])])

    baseline = float(np.median(radii[valid]))
    half_level = baseline + 0.5 * (radii[peak] - baseline)
    left = peak
    while left - 1 >= 0 and radii[left - 1] >= half_level:
        left -= 1
    right = peak
    while right + 1 < n_bins and radii[right + 1] >= half_level:
        right += 1

    x_lo, x_hi = float(edges[left]), float(edges[right + 1])
    return {
        "axial_dim": axial,
        "x_lo": x_lo, "x_hi": x_hi,
        "x_center": float(centers[peak]),
        "half_width": 0.5 * (x_hi - x_lo),
        "center2d": cen2d[peak].tolist(),
        "radius": float(radii[peak]),
    }
