"""Validation metrics: PINN surrogate vs CFD ground truth.

Compares predicted velocity (on any CFD slice/volume) and wall shear stress
against the CFD data, and writes machine-readable (JSON, CSV) plus
human-readable (TXT) summaries into ``report/metrics``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..config import METRICS_DIR, MODELS_DIR, PROJECT_ROOT, TABLES_DIR
from ..data.cache import load_points
from ..data.geometry import compute_wall_normals, sac_axial_band
from ..data.registry import CaseRecord, cases_by_id, load_registry
from .predict import TrainedModel, _phase_value, load_trained, predict_physical


def _rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))


def _pearson(pred: np.ndarray, true: np.ndarray) -> float:
    """Pearson correlation; NaN for <2 points or a constant input."""
    a, b = np.asarray(pred, float).ravel(), np.asarray(true, float).ravel()
    if a.size < 2:
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _spearman(pred: np.ndarray, true: np.ndarray) -> float:
    """Spearman rank correlation (Pearson of ranks); NaN for <2 points."""
    a, b = np.asarray(pred, float).ravel(), np.asarray(true, float).ravel()
    if a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return _pearson(ra, rb)


def _ccc(pred: np.ndarray, true: np.ndarray) -> float:
    """Lin's concordance correlation coefficient (agreement, not just correlation).

    Penalizes a constant offset/scale that a Pearson r would miss, so it catches
    the jet-core over-prediction (a slope>1 bias) that correlation alone hides.
    """
    a, b = np.asarray(pred, float).ravel(), np.asarray(true, float).ravel()
    if a.size < 2:
        return float("nan")
    cov = np.mean((a - a.mean()) * (b - b.mean()))
    denom = a.var() + b.var() + (a.mean() - b.mean()) ** 2
    return float(2.0 * cov / denom) if denom > 1e-12 else float("nan")


def _bland_altman(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """Bland-Altman agreement of ``pred`` against ``true`` (C2).

    Bias is the mean signed difference (pred-true), the limits of agreement are
    bias +/- 1.96*SD of the difference, and ``bias_pct`` expresses the bias as a
    percentage of the mean true magnitude. Surfaces directional WSS bias (e.g. the
    positive bias at high WSS from peak over-prediction) that an RMS metric hides.
    """
    a, b = np.asarray(pred, float).ravel(), np.asarray(true, float).ravel()
    diff = a - b
    bias, sd = float(diff.mean()), float(diff.std())
    mean_true = float(b.mean())
    return {
        "ba_bias": bias, "ba_sd": sd,
        "ba_loa_low": bias - 1.96 * sd, "ba_loa_high": bias + 1.96 * sd,
        "ba_bias_pct": float(100.0 * bias / mean_true) if abs(mean_true) > 1e-9 else float("nan"),
    }


def _dominant_direction(uvw: np.ndarray) -> np.ndarray:
    m = uvw.mean(axis=0)
    n = np.linalg.norm(m)
    return m / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])


def recirculation_fraction(uvw: np.ndarray, direction: Optional[np.ndarray] = None,
                           speed_floor: float = 0.0) -> float:
    """Fraction of samples whose velocity opposes the dominant flow ``direction``.

    M4: two robustness fixes over the naive version.
      * ``speed_floor`` excludes near-stagnant points (speed below the floor). In
        diastole most of the slice is ~0, so the *sign* of those vectors is pure
        noise; without a floor the recirculation fraction is noise-dominated and a
        ~exact diastolic field reads as wildly different from CFD.
      * ``direction`` is supplied (the CFD-derived axis) so the SAME reference is
        used for both CFD and PINN, making the two fractions directly comparable;
        a per-field mean direction would shift between them.

    Returns NaN when no sample clears ``speed_floor``: there is then no flow to take
    a fraction of, and NaN keeps "no data" distinguishable from a measured 0.0 ("flow
    present, none of it reversed").
    """
    speed = np.linalg.norm(uvw, axis=1)
    mask = speed > speed_floor
    if not mask.any():
        return float("nan")
    v = uvw[mask]
    d = _dominant_direction(v) if direction is None else np.asarray(direction, float)
    return float(np.mean(v @ d < 0.0))


def secondary_flow_fraction(uvw: np.ndarray, direction: Optional[np.ndarray] = None,
                            speed_floor: float = 0.0) -> float:
    """Mean transverse (out-of-axis) fraction of velocity over significant points.

    For each vector ``v`` with shared axial unit direction ``d``, the secondary
    (swirl / cross-flow) fraction is ``|v - (v·d)d| / |v|``: pure axial flow -> 0,
    pure transverse -> 1. Unlike helicity it needs no velocity gradients, so it is
    computed identically -- and therefore comparably -- on scattered CFD samples and
    PINN predictions. Supply the SAME CFD-derived ``direction`` to both fields (as
    ``recirculation_fraction`` does) so the two fractions are directly comparable.

    Returns NaN when no sample clears ``speed_floor`` (no flow to characterize), so
    "no data" stays distinguishable from a measured 0.0 (purely axial flow).
    """
    uvw = np.asarray(uvw, float)
    speed = np.linalg.norm(uvw, axis=1)
    mask = speed > speed_floor
    if not mask.any():
        return float("nan")
    v = uvw[mask]
    s = speed[mask]
    d = _dominant_direction(v) if direction is None else np.asarray(direction, float)
    perp = v - np.outer(v @ d, d)
    return float(np.mean(np.linalg.norm(perp, axis=1) / s))


def velocity_metrics(model: TrainedModel, records: Sequence[CaseRecord], case_id: int,
                     phase: str, kind: str = "XZ", max_points: int = 60_000) -> Optional[Dict]:
    """Velocity error of the surrogate on a CFD slice (default XZ plane)."""
    rec = cases_by_id(records)[case_id]
    df = load_points(rec, kind, phase)
    if df is None or not {"u", "v", "w"}.issubset(df.columns):
        return None
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)
    coords = df[["x", "y", "z"]].to_numpy(float)
    true = df[["u", "v", "w"]].to_numpy(float)
    beta = rec.beta if rec.beta is not None else 1.0
    pred = predict_physical(model, coords, rec.inlet_diameter_cm, rec.disease_flag, phase, beta=beta)
    pred_v = np.column_stack([pred["u"], pred["v"], pred["w"]])
    true_speed = np.linalg.norm(true, axis=1)
    # RMSE normalized by the global reference velocity U_ref. Unlike rel-L2 (which
    # divides by the field's own norm), this stays interpretable for near-stagnant
    # phases such as diastole, where ||true|| -> 0 makes rel-L2 blow up.
    u_ref = model.normalizer.U_ref
    vel_nrmse_uref = float(np.sqrt(np.mean((pred_v - true) ** 2)) / u_ref)
    speed_nrmse_uref = float(np.sqrt(np.mean((pred["speed"] - true_speed) ** 2)) / u_ref)
    # Q1: per-PHASE NRMSE, normalized by THIS phase's own speed scale (0.995-quantile
    # of the phase's true speed) instead of the global (systolic-scale) U_ref. The
    # global-U_ref NRMSE hides diastolic collapse (a 12x over-prediction reads as a
    # benign few-percent); the per-phase NRMSE exposes it. This is the honest metric.
    # Guard against a degenerate slice: some CFD plane exports carry no resolved
    # flow (e.g. Case 3 systolic XY/XZ are all-zero). There the phase scale collapses
    # to the floor and per-phase NRMSE / rel-L2 explode to meaningless ~1e6 values.
    # Detect it and report NaN for the phase-normalized metrics rather than a number
    # that would wreck any aggregate, while keeping the U_ref-normalized metric.
    q995 = float(np.quantile(true_speed, 0.995))
    degenerate = q995 < 1e-4          # no resolved flow on this slice
    phase_u_ref = max(q995, 1e-6)
    nan = float("nan")
    vel_nrmse_phase = nan if degenerate else float(np.sqrt(np.mean((pred_v - true) ** 2)) / phase_u_ref)
    speed_nrmse_phase = nan if degenerate else float(
        np.sqrt(np.mean((pred["speed"] - true_speed) ** 2)) / phase_u_ref)
    # B1 continuity residual (independent of the CFD field, so always computed);
    # B2 speed agreement (NaN on a degenerate slice where the CFD field is empty).
    div = predicted_divergence(model, coords, rec.inlet_diameter_cm, rec.disease_flag,
                               phase, beta=beta)
    return {
        "case": case_id, "phase": phase, "kind": kind, "n_points": int(len(df)),
        "degenerate_slice": bool(degenerate),
        "vel_rel_l2": nan if degenerate else _rel_l2(pred_v, true),
        "speed_rel_l2": nan if degenerate else _rel_l2(pred["speed"], true_speed),
        "vel_nrmse_uref": vel_nrmse_uref,
        "speed_nrmse_uref": speed_nrmse_uref,
        "phase_u_ref": phase_u_ref,
        "vel_nrmse_phase": vel_nrmse_phase,
        "speed_nrmse_phase": speed_nrmse_phase,
        "speed_pearson_r": nan if degenerate else _pearson(pred["speed"], true_speed),
        "speed_ccc": nan if degenerate else _ccc(pred["speed"], true_speed),
        "div_rms": div["div_rms"], "div_mean_abs": div["div_mean_abs"],
        "div_rel": div["div_rel"],
        # M4: floor at 5% of the phase's own peak speed; reference direction from
        # the CFD field's significant-speed points, shared by both fractions.
        # NaN on a degenerate slice (no CFD field to derive the axis/floor from).
        **_flow_pair(true, pred_v, phase_u_ref, degenerate=degenerate),
        # Sac-masked recirc/swirl (diseased cases only): isolates the bulge so the
        # through-flow does not dilute the recirculation signal. NaN otherwise.
        **_sac_flow_metrics(rec, phase, coords, true, pred_v, phase_u_ref,
                            degenerate=degenerate),
    }


def _flow_pair(true: np.ndarray, pred: np.ndarray, phase_u_ref: float,
               suffix: str = "", degenerate: bool = False) -> Dict[str, float]:
    """Recirculation + secondary-flow (swirl) fractions for CFD vs PINN.

    Both quantities use the SAME CFD-derived axis and speed floor for the two fields
    so they are directly comparable. ``suffix`` namespaces the keys (e.g. ``_sac``).

    ``degenerate`` (a CFD slice with no resolved flow) yields NaN for all four keys,
    exactly as the other CFD-comparison metrics do. Both the axis and the floor are
    derived from the CFD field, so with no CFD field there is nothing to compare
    against: the floor would collapse to ~5e-8, admitting pure noise, and the CFD
    fractions would read a vacuous 0.0 against a fully-populated PINN fraction —
    a large apparent disagreement that is an artifact of the empty slice, not physics.
    """
    nan = float("nan")
    if degenerate:
        return {f"recirc_cfd{suffix}": nan, f"recirc_pinn{suffix}": nan,
                f"swirl_cfd{suffix}": nan, f"swirl_pinn{suffix}": nan}
    floor = 0.05 * phase_u_ref
    true_speed = np.linalg.norm(true, axis=1)
    sig = true[true_speed > floor]
    direction = _dominant_direction(sig if len(sig) else true)
    return {
        f"recirc_cfd{suffix}": recirculation_fraction(true, direction, floor),
        f"recirc_pinn{suffix}": recirculation_fraction(pred, direction, floor),
        f"swirl_cfd{suffix}": secondary_flow_fraction(true, direction, floor),
        f"swirl_pinn{suffix}": secondary_flow_fraction(pred, direction, floor),
    }


def _sac_flow_metrics(rec: CaseRecord, phase: str, coords: np.ndarray, true: np.ndarray,
                      pred: np.ndarray, phase_u_ref: float,
                      degenerate: bool = False) -> Dict[str, float]:
    """Recirc/swirl restricted to the saccular bulge band (diseased cases only).

    Locates the bulge from the wall point cloud (``sac_axial_band``) and keeps only
    the plane-slice points whose axial coordinate falls inside it. Healthy cases (no
    interior bulge), cases without a wall cloud / enough in-band points, and
    degenerate slices (no resolved CFD flow) return NaN so they neither claim a sac
    measurement nor wreck any aggregate.
    """
    nan = float("nan")
    blank = {k: nan for k in ("recirc_cfd_sac", "recirc_pinn_sac",
                              "swirl_cfd_sac", "swirl_pinn_sac")}
    if degenerate or int(rec.disease_flag) != 1:
        return blank
    wall = load_points(rec, "WSS", phase)
    if wall is None or not {"x", "y", "z"}.issubset(wall.columns):
        return blank
    band = sac_axial_band(wall[["x", "y", "z"]].to_numpy(float))
    ax = band["axial_dim"]
    in_band = (coords[:, ax] >= band["x_lo"]) & (coords[:, ax] <= band["x_hi"])
    if int(in_band.sum()) < 20:
        return blank
    return _flow_pair(true[in_band], pred[in_band], phase_u_ref, suffix="_sac",
                      degenerate=degenerate)


def wss_metrics(model: TrainedModel, records: Sequence[CaseRecord], case_id: int,
                phase: str, max_points: int = 20_000) -> Optional[Dict]:
    """WSS-magnitude error of the surrogate on the wall point cloud."""
    rec = cases_by_id(records)[case_id]
    df = load_points(rec, "WSS", phase)
    if df is None or "wss" not in df.columns:
        return None
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)
    coords = df[["x", "y", "z"]].to_numpy(float)
    true_mag = df["wss"].to_numpy(float)
    normals = compute_wall_normals(coords)
    beta = rec.beta if rec.beta is not None else 1.0
    pred = predict_wss_physical(model, coords, normals, rec.inlet_diameter_cm,
                                rec.disease_flag, phase, beta=beta)
    pm = pred["wss_magnitude"]
    # C1: per-phase WSS NRMSE normalized by this phase's own 0.995-quantile WSS.
    # PRIMARY WSS metric: rel-L2 on the sub-1-Pa diastolic field is dominated by
    # tiny values and not comparable to the ~13-17 Pa systolic field.
    wss_scale = max(float(np.quantile(true_mag, 0.995)), 1e-6)
    wss_nrmse = float(np.sqrt(np.mean((pm - true_mag) ** 2)) / wss_scale)
    ba = _bland_altman(pm, true_mag)            # C2
    return {
        "case": case_id, "phase": phase, "n_points": int(len(df)),
        "wss_rel_l2": _rel_l2(pm, true_mag),
        "wss_nrmse": wss_nrmse, "wss_scale": wss_scale,
        "wss_mean_cfd": float(true_mag.mean()), "wss_mean_pinn": float(pm.mean()),
        "wss_peak_cfd": float(true_mag.max()), "wss_peak_pinn": float(pm.max()),
        "wss_ba_bias": ba["ba_bias"], "wss_ba_sd": ba["ba_sd"],
        "wss_ba_loa_low": ba["ba_loa_low"], "wss_ba_loa_high": ba["ba_loa_high"],
        "wss_ba_bias_pct": ba["ba_bias_pct"],
        "wss_pearson_r": _pearson(pm, true_mag), "wss_ccc": _ccc(pm, true_mag),
    }


def pressure_metrics(model: TrainedModel, records: Sequence[CaseRecord], case_id: int,
                     phase: str, max_points: int = 20_000) -> Optional[Dict]:
    """Wall-pressure pattern agreement (D1/D2).

    The CFX/Fluent pressure datum is arbitrary and exported wall-only, so only the
    spatial pattern is meaningful: both fields are mean-removed and compared by rank
    (Spearman) and linear (Pearson) correlation plus a robust ``(q0.99-q0.01)`` range
    ratio. Absolute pressure is deliberately not scored.
    """
    rec = cases_by_id(records)[case_id]
    df = load_points(rec, "WSS", phase)
    if df is None or "p" not in df.columns:
        return None
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)
    coords = df[["x", "y", "z"]].to_numpy(float)
    true_p = df["p"].to_numpy(float)
    beta = rec.beta if rec.beta is not None else 1.0
    pred_p = predict_physical(model, coords, rec.inlet_diameter_cm, rec.disease_flag,
                              phase, beta=beta)["p"]
    tp = true_p - true_p.mean()
    pp = pred_p - pred_p.mean()

    def _range(x: np.ndarray) -> float:
        return float(np.quantile(x, 0.99) - np.quantile(x, 0.01))

    rng_cfd, rng_pinn = _range(true_p), _range(pred_p)
    return {
        "case": case_id, "phase": phase, "n_points": int(len(df)),
        "dp_pearson_r": _pearson(pp, tp),
        "dp_spearman_r": _spearman(pp, tp),
        "dp_rel_l2": _rel_l2(pp, tp),
        "dp_range_cfd": rng_cfd, "dp_range_pinn": rng_pinn,
        "dp_range_ratio": (rng_pinn / rng_cfd) if rng_cfd > 1e-9 else float("nan"),
    }


def _json_safe(value):
    """NaN/Inf -> None so the emitted file is valid JSON.

    Non-finite floats are expected here (degenerate slices, healthy-case sac
    metrics), but ``json.dumps`` would write the bare tokens ``NaN``/``Infinity``,
    which Python reads back yet strict parsers (jq, JavaScript ``JSON.parse``)
    reject. ``null`` round-trips to ``None``, which the readers already handle.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_report(rows: List[Dict], name: str, title: str = "", out_dir: Path = METRICS_DIR) -> Path:
    """Write metrics to JSON + CSV + a human-readable TXT under ``out_dir``.

    In the JSON, NaN/Inf are written as ``null`` (see ``_json_safe``); the CSV/TXT
    keep their literal ``nan`` rendering for readability.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in rows if r is not None]
    safe = [{k: _json_safe(v) for k, v in r.items()} for r in rows]
    (out_dir / f"{name}.json").write_text(json.dumps(safe, indent=2), encoding="utf-8")

    if rows:
        keys = list(rows[0].keys())
        with open(out_dir / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    lines = [title or name, "=" * len(title or name), ""]
    for r in rows:
        head = f"case {r.get('case')} {r.get('phase','')}".strip()
        body = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in r.items() if k not in ("case", "phase"))
        lines.append(f"{head:>18} | {body}")
    txt = out_dir / f"{name}.txt"
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return txt


def predict_wss_physical(model: TrainedModel, coords_phys: np.ndarray, normals: np.ndarray,
                         diameter_cm: float, disease_flag: int, phase: str,
                         beta: float = 1.0, batch: int = 20_000) -> Dict[str, np.ndarray]:
    """Predict wall shear stress (Pa) at wall points with inward ``normals``.

    Newtonian non-dimensional wall stress equals the standardized strain rate, so
    physical WSS is the tangential traction times ``tau_ref = mu * U_ref / L``.
    """
    norm = model.normalizer
    cs = norm.coords_std(coords_phys).astype(np.float32)
    mu = np.array([norm.diameter_nd(diameter_cm), float(beta), float(disease_flag),
                   _phase_value(phase)], dtype=np.float32)
    nets = model.networks

    out = []
    for i in range(0, len(cs), batch):
        c = cs[i:i + batch]
        nrm = torch.tensor(normals[i:i + batch], dtype=torch.float32, device=model.device)
        params = torch.tensor(np.tile(mu, (len(c), 1)), device=model.device)
        x = torch.tensor(c[:, 0:1], device=model.device, requires_grad=True)
        y = torch.tensor(c[:, 1:2], device=model.device, requires_grad=True)
        z = torch.tensor(c[:, 2:3], device=model.device, requires_grad=True)
        net_in = torch.cat([x, y, z, params], dim=1)
        u = nets["u"](net_in).view(-1, 1)
        v = nets["v"](net_in).view(-1, 1)
        w = nets["w"](net_in).view(-1, 1)
        one = torch.ones_like(u)

        def g(o, i_):
            return torch.autograd.grad(o, i_, grad_outputs=one, create_graph=False,
                                       retain_graph=True)[0]

        ux, uy, uz = g(u, x), g(u, y), g(u, z)
        vx, vy, vz = g(v, x), g(v, y), g(v, z)
        wx, wy, wz = g(w, x), g(w, y), g(w, z)

        txx, tyy, tzz = 2 * ux, 2 * vy, 2 * wz
        txy, txz, tyz = (uy + vx), (uz + wx), (vz + wy)
        nx, ny, nz = nrm[:, 0:1], nrm[:, 1:2], nrm[:, 2:3]
        tx = txx * nx + txy * ny + txz * nz
        ty = txy * nx + tyy * ny + tyz * nz
        tz = txz * nx + tyz * ny + tzz * nz
        tdn = tx * nx + ty * ny + tz * nz
        wss = torch.cat([tx - tdn * nx, ty - tdn * ny, tz - tdn * nz], dim=1)
        out.append(wss.detach().cpu().numpy() * norm.tau_ref)  # -> physical Pa

    comp = np.vstack(out)
    return {"wss_components": comp, "wss_magnitude": np.linalg.norm(comp, axis=1)}


def predicted_divergence(model: TrainedModel, coords_phys: np.ndarray, diameter_cm: float,
                         disease_flag: int, phase: str, beta: float = 1.0,
                         batch: int = 20_000) -> Dict[str, float]:
    """Continuity residual of the predicted velocity field (B1).

    Computes the divergence and the velocity-gradient Frobenius norm in
    STANDARDIZED coordinates by autograd (the nets take standardized coords and
    emit standardized velocity, so the constant per-phase gain cancels in the
    ratio). Reports ``div_rel = rms(div u) / rms(||grad u||_F)`` -- a dimensionless,
    scale-free measure of how well incompressibility is satisfied, framed as a soft
    residual rather than an exact constraint. Mirrors the autograd assembly in
    ``predict_wss_physical``.
    """
    norm = model.normalizer
    cs = norm.coords_std(coords_phys).astype(np.float32)
    mu = np.array([norm.diameter_nd(diameter_cm), float(beta), float(disease_flag),
                   _phase_value(phase)], dtype=np.float32)
    nets = model.networks

    div_sq = div_abs = grad_sq = 0.0
    n = 0
    for i in range(0, len(cs), batch):
        c = cs[i:i + batch]
        params = torch.tensor(np.tile(mu, (len(c), 1)), device=model.device)
        x = torch.tensor(c[:, 0:1], device=model.device, requires_grad=True)
        y = torch.tensor(c[:, 1:2], device=model.device, requires_grad=True)
        z = torch.tensor(c[:, 2:3], device=model.device, requires_grad=True)
        net_in = torch.cat([x, y, z, params], dim=1)
        u = nets["u"](net_in).view(-1, 1)
        v = nets["v"](net_in).view(-1, 1)
        w = nets["w"](net_in).view(-1, 1)
        one = torch.ones_like(u)

        def g(o, i_):
            return torch.autograd.grad(o, i_, grad_outputs=one, create_graph=False,
                                       retain_graph=True)[0]

        ux, uy, uz = g(u, x), g(u, y), g(u, z)
        vx, vy, vz = g(v, x), g(v, y), g(v, z)
        wx, wy, wz = g(w, x), g(w, y), g(w, z)
        d = (ux + vy + wz).detach()
        gf = (ux ** 2 + uy ** 2 + uz ** 2 + vx ** 2 + vy ** 2 + vz ** 2
              + wx ** 2 + wy ** 2 + wz ** 2).detach()
        div_sq += float((d ** 2).sum())
        div_abs += float(d.abs().sum())
        grad_sq += float(gf.sum())
        n += len(c)

    if n == 0:
        nan = float("nan")
        return {"div_rms": nan, "div_mean_abs": nan, "div_rel": nan}
    div_rms = (div_sq / n) ** 0.5
    grad_rms = (grad_sq / n) ** 0.5
    return {
        "div_rms": float(div_rms),
        "div_mean_abs": float(div_abs / n),
        "div_rel": float(div_rms / (grad_rms + 1e-12)),
    }


# ---------------------------------------------------------------------------
# Cross-run reporting: evaluate every trained checkpoint, and the LODO table
# ---------------------------------------------------------------------------
# Both are pure post-processing over runs that already exist (the evaluation reloads
# checkpoints for inference; the table reads only JSON). Driven by scripts/report.py.
#
# A caveat that governs the LODO table: each fold fits its own train-set
# nondimensionalization, so only scale-free errors (NRMSE, relative L2, ratios) are
# comparable across folds. Never average 2.0/2.3/2.6 into a single "generalization"
# number -- 2.3 cm interpolates between the training diameters while 2.0 and 2.6 cm
# extrapolate past a single bracketing one, and they behave differently.

PHASES = ("systolic", "diastolic")

_SYM = {"axisymmetric": "axisym.", "anterior": "anterior", "posterior": "posterior",
        "healthy": "healthy"}
_PHASE_ORDER = {"systolic": 0, "diastolic": 1}

# A run may sit directly under models/<name>/ (the default output_dir) or be filed into
# a models/<group>/<name>/ subfolder once a sweep is grouped; check both.
_MODEL_SUBDIRS = ("", "_insample", "_ablation", "_seedsweep")

# (label, experiment name, [cases], regime, config that produces the checkpoint)
# In-sample rows are the per-case reconstruction floor; LODO rows are the three
# leave-one-diameter-out folds, each evaluated on the cases its fold held out.
REPORTED_RUNS = [
    ("insample_c1", "stageA_case1_s12", [1], "in-sample", "configs/stageA_case1_s12.yaml"),
    ("insample_c2", "stageA_case2_insample", [2], "in-sample", "configs/stageA_case2_insample.yaml"),
    ("insample_c3", "stageA_case3_insample", [3], "in-sample", "configs/stageA_case3_insample.yaml"),
    ("insample_c4", "stageA_case4_insample", [4], "in-sample", "configs/stageA_case4_insample.yaml"),
    ("insample_c7", "stageA_case7_insample", [7], "in-sample", "configs/stageA_case7_insample.yaml"),
    ("lodo_2.0", "stageB_kfold_hold2p0", [1, 2, 3], "LODO-2.0", "configs/stageB_kfold_hold2p0.yaml"),
    ("lodo_2.3", "stageB_richerloo_f16", [4, 5, 6], "LODO-2.3", "configs/stageB_richerloo_f16.yaml"),
    ("lodo_2.6", "stageB_kfold_hold2p6", [7, 8, 9], "LODO-2.6", "configs/stageB_kfold_hold2p6.yaml"),
]


def find_checkpoint(name: str, filename: str = "best_model.pt") -> Optional[Path]:
    """Locate a run's checkpoint by experiment name, searching the known groupings."""
    for sub in _MODEL_SUBDIRS:
        base = (MODELS_DIR / sub / name) if sub else (MODELS_DIR / name)
        ck = base / filename
        if ck.exists():
            return ck
    return None


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def _mean_of(xs) -> float:
    v = [x for x in xs if _is_num(x)]
    return sum(v) / len(v) if v else float("nan")


def _fmt(x, p: int = 3) -> str:
    """Format a metric for display; missing/NaN renders as an em dash."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else str(x)


def load_metric_file(exp: str, kind: str) -> Dict:
    """Read ``report/metrics/<exp>/<kind>.json`` keyed by (case, phase); {} if absent."""
    p = METRICS_DIR / exp / f"{kind}.json"
    if not p.exists():
        return {}
    return {(r["case"], r["phase"]): r for r in json.loads(p.read_text())}


def _fold_mean(exp: str, kind: str, key: str) -> float:
    """Mean of one metric over every row of one experiment's metric file."""
    p = METRICS_DIR / exp / f"{kind}.json"
    if not p.exists():
        return float("nan")
    return _mean_of(r.get(key) for r in json.loads(p.read_text()))


def parse_folds(specs: Sequence[str]) -> List:
    """Parse ``"2.3:experiment_name"`` fold specs into (held_diameter_cm, experiment)."""
    out = []
    for spec in specs:
        d, _, exp = spec.partition(":")
        if not exp:
            raise ValueError(f"bad fold spec {spec!r}; expected 'held_cm:experiment'")
        out.append((float(d), exp))
    return out


def interp_diameter(diams: Sequence[float]) -> Optional[float]:
    """The interpolated (middle) diameter, or None when the folds do not bracket one."""
    return sorted(diams)[len(diams) // 2] if len(diams) >= 3 else None


def _g(d: Optional[Dict], k: str):
    return d.get(k) if d else None


def _f6(x) -> str:
    if x is None:
        return "  --  "
    try:
        if x != x:  # NaN
            return "  nan "
        return f"{x:6.3f}"
    except (TypeError, ValueError):
        return str(x)


def evaluate_runs(runs: Optional[Sequence[str]] = None, device: str = "cuda",
                  out: Optional[Path] = None) -> Path:
    """Evaluate every reported run against CFD, straight from its saved checkpoint.

    Recomputes the velocity, WSS, pressure and continuity metrics for each
    (run, case, phase) and writes them as one JSON table, so the numbers quoted in
    the paper can be regenerated in a single command without retraining.

    Inference only -- no training. A run whose checkpoint does not exist yet is
    reported as ``[skip]`` with the config that would produce it, rather than
    failing the whole batch.
    """
    selected = REPORTED_RUNS if not runs else [r for r in REPORTED_RUNS if r[0] in set(runs)]
    records = load_registry()
    rows: List[Dict] = []
    for label, name, cases, regime, cfg_path in selected:
        ck = find_checkpoint(name)
        if ck is None:
            print(f"[skip] {label}: no checkpoint for '{name}' under {MODELS_DIR}. "
                  f"Train it first:  python scripts/run.py --config {cfg_path}")
            continue
        model = load_trained(ck, device=device)
        for cid in cases:
            for ph in PHASES:
                vm = velocity_metrics(model, records, cid, ph, kind="XZ")
                wm = wss_metrics(model, records, cid, ph)
                pm = pressure_metrics(model, records, cid, ph)
                peak_ratio = None
                if wm and wm["wss_peak_cfd"] > 1e-9:
                    peak_ratio = wm["wss_peak_pinn"] / wm["wss_peak_cfd"]
                rows.append({
                    "label": label, "regime": regime, "case": cid, "phase": ph,
                    "vel_nrmse_phase": _g(vm, "vel_nrmse_phase"),
                    "vel_rel_l2": _g(vm, "vel_rel_l2"),
                    "div_rel": _g(vm, "div_rel"), "div_rms": _g(vm, "div_rms"),
                    "recirc_cfd": _g(vm, "recirc_cfd"), "recirc_pinn": _g(vm, "recirc_pinn"),
                    "wss_nrmse": _g(wm, "wss_nrmse"), "wss_rel_l2": _g(wm, "wss_rel_l2"),
                    "wss_peak_cfd": _g(wm, "wss_peak_cfd"), "wss_peak_pinn": _g(wm, "wss_peak_pinn"),
                    "wss_peak_ratio": peak_ratio, "wss_pearson_r": _g(wm, "wss_pearson_r"),
                    "dp_pearson_r": _g(pm, "dp_pearson_r"), "dp_spearman_r": _g(pm, "dp_spearman_r"),
                    "dp_range_ratio": _g(pm, "dp_range_ratio"),
                })
                r = rows[-1]
                print(f"{label:12s} c{cid} {ph:9s} "
                      f"velNRMSE={_f6(r['vel_nrmse_phase'])} div_rel={_f6(r['div_rel'])} "
                      f"wssNRMSE={_f6(r['wss_nrmse'])} peakR={_f6(r['wss_peak_ratio'])} "
                      f"dp_r={_f6(r['dp_pearson_r'])} dp_range={_f6(r['dp_range_ratio'])}")

    out = Path(out) if out is not None else (METRICS_DIR / "all_runs.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([{k: _json_safe(v) for k, v in r.items()} for r in rows],
                              indent=2), encoding="utf-8")
    print(f"\n[evaluate] wrote {out}  ({len(rows)} rows)")
    return out


def kfold_table(folds: Sequence[str], phases: Sequence[str] = PHASES,
                out: str = "lodo_kfold",
                title: str = "Leave-one-diameter-out generalization (per-phase NRMSE)"
                ) -> Optional[Path]:
    """Build the LODO table from each fold's metric JSON; write Markdown + CSV + LaTeX.

    Emits per-(case, phase) rows plus a per-fold mean, labelling every fold as
    interpolation (the middle held diameter) or extrapolation.
    """
    parsed = parse_folds(folds)
    by = cases_by_id(load_registry())
    mid = interp_diameter([d for d, _ in parsed])

    rows: List[Dict] = []
    for d, exp in parsed:
        nature = "interp" if (mid is not None and abs(d - mid) < 1e-9) else "extrap"
        vel, wss = load_metric_file(exp, "velocity"), load_metric_file(exp, "wss")
        keys = sorted(set(vel) | set(wss), key=lambda k: (k[0], _PHASE_ORDER.get(k[1], 9)))
        fold_vel, fold_wss = [], []
        for case, phase in keys:
            if phase not in phases:
                continue
            v, w = vel.get((case, phase), {}), wss.get((case, phase), {})
            pc, pp = w.get("wss_peak_cfd"), w.get("wss_peak_pinn")
            rows.append({
                "held_cm": d, "nature": nature, "case": case,
                "symmetry": _SYM.get(getattr(by.get(case), "symmetry", ""), "?"),
                "phase": phase,
                "vel_nrmse": v.get("vel_nrmse_phase"), "wss_nrmse": w.get("wss_nrmse"),
                "peak_ratio": (pp / pc) if (pc and pp) else None,
                "recirc_cfd": v.get("recirc_cfd"), "recirc_pinn": v.get("recirc_pinn"),
                "div_rel": v.get("div_rel"),
            })
            fold_vel.append(rows[-1]["vel_nrmse"])
            fold_wss.append(rows[-1]["wss_nrmse"])
        rows.append({
            "held_cm": d, "nature": nature, "case": "mean", "symmetry": "", "phase": "",
            "vel_nrmse": _mean_of(fold_vel), "wss_nrmse": _mean_of(fold_wss),
            "peak_ratio": None, "recirc_cfd": None, "recirc_pinn": None, "div_rel": None,
        })

    if not rows:
        print("[kfold] no metrics found for the requested folds.")
        return None

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["held_cm", "nature", "case", "symmetry", "phase", "vel_nrmse", "wss_nrmse",
            "peak_ratio", "recirc_cfd", "recirc_pinn", "div_rel"]
    with open(TABLES_DIR / f"{out}.csv", "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wri.writeheader()
        wri.writerows(rows)

    def recirc_cell(r, sep=" / "):
        if r["recirc_cfd"] is None:
            return "—"
        return f"{_fmt(r['recirc_cfd'])}{sep}{_fmt(r['recirc_pinn'])}"

    hdr = ["Held (cm)", "Nature", "Case (sym)", "Phase", "Vel NRMSE", "WSS NRMSE",
           "Peak P/C", "Recirc CFD/PINN", "div_rel"]
    md = [f"# {title}", "",
          "| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        case = "**mean**" if r["case"] == "mean" else f"{r['case']} ({r['symmetry']})"
        md.append("| " + " | ".join([
            _fmt(r["held_cm"], 1), r["nature"], case, r["phase"],
            _fmt(r["vel_nrmse"]), _fmt(r["wss_nrmse"]), _fmt(r["peak_ratio"], 2),
            recirc_cell(r), _fmt(r["div_rel"]),
        ]) + " |")
    md_path = TABLES_DIR / f"{out}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = [r"\begin{table}[t]", r"  \centering",
           rf"  \caption{{{title}. 2.3 cm is interpolation; 2.0/2.6 cm are "
           r"extrapolation (single bracketing diameter). Each fold uses its own "
           r"train-set scales; do not average across folds.}}",
           r"  \label{tab:lodo}", r"  \small", r"  \begin{tabular}{llllccccc}",
           r"    \toprule",
           r"    Held & Nature & Case (sym.) & Phase & Vel NRMSE & WSS NRMSE & Peak "
           r"& Recirc C/P & $\mathrm{div}_{\mathrm{rel}}$ \\", r"    \midrule"]
    for r in rows:
        case = r"\textbf{mean}" if r["case"] == "mean" else f"{r['case']} ({r['symmetry']})"
        tex.append("    " + " & ".join([
            _fmt(r["held_cm"], 1), r["nature"], case, r["phase"], _fmt(r["vel_nrmse"]),
            _fmt(r["wss_nrmse"]), _fmt(r["peak_ratio"], 2), recirc_cell(r, "/"),
            _fmt(r["div_rel"]),
        ]) + r" \\")
        if r["case"] == "mean":
            tex.append(r"    \midrule")
    tex += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (TABLES_DIR / f"{out}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    print(f"[kfold] wrote {md_path.relative_to(PROJECT_ROOT)} (+ .csv, .tex), {len(rows)} rows")
    print("\n".join(md))
    return md_path
