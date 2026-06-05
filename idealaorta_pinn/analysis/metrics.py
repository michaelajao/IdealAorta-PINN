"""Validation metrics: PINN surrogate vs CFD ground truth.

Compares predicted velocity (on any CFD slice/volume) and wall shear stress
against the CFD data, and writes machine-readable (JSON, CSV) plus
human-readable (TXT) summaries into ``report/metrics``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..config import METRICS_DIR
from ..data.cache import load_points
from ..data.geometry import compute_wall_normals
from ..data.registry import CaseRecord, cases_by_id
from .predict import TrainedModel, _phase_value, predict_physical


def _rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))


def recirculation_fraction(uvw: np.ndarray) -> float:
    """Fraction of samples whose velocity opposes the dominant flow direction."""
    mean_vec = uvw.mean(axis=0)
    n = np.linalg.norm(mean_vec)
    direction = mean_vec / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])
    return float(np.mean(uvw @ direction < 0.0))


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
    return {
        "case": case_id, "phase": phase, "kind": kind, "n_points": int(len(df)),
        "vel_rel_l2": _rel_l2(pred_v, true),
        "speed_rel_l2": _rel_l2(pred["speed"], true_speed),
        "recirc_cfd": recirculation_fraction(true),
        "recirc_pinn": recirculation_fraction(pred_v),
    }


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
    return {
        "case": case_id, "phase": phase, "n_points": int(len(df)),
        "wss_rel_l2": _rel_l2(pm, true_mag),
        "wss_mean_cfd": float(true_mag.mean()), "wss_mean_pinn": float(pm.mean()),
        "wss_peak_cfd": float(true_mag.max()), "wss_peak_pinn": float(pm.max()),
    }


def write_report(rows: List[Dict], name: str, title: str = "") -> Path:
    """Write metrics to JSON + CSV + a human-readable TXT under report/metrics."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [r for r in rows if r is not None]
    (METRICS_DIR / f"{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if rows:
        keys = list(rows[0].keys())
        with open(METRICS_DIR / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    lines = [title or name, "=" * len(title or name), ""]
    for r in rows:
        head = f"case {r.get('case')} {r.get('phase','')}".strip()
        body = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in r.items() if k not in ("case", "phase"))
        lines.append(f"{head:>18} | {body}")
    txt = METRICS_DIR / f"{name}.txt"
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
