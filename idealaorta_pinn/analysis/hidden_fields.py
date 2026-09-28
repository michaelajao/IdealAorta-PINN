"""Can a velocity-only PINN infer the fields it was never shown? WSS and pressure, Case 1.

Compares, on the whole wall and the aneurysm wall zone:

    PINN          wall shear from the network's own velocity gradients; pressure from its
                  pressure net (only an outlet-face datum was given)
    interpolation the same training velocity (even cubes + zero-velocity wall), linearly
                  interpolated; WSS from the tangential velocity a distance h inside the wall
                  along the inward normal, first order mu*u_t(h)/h and second order
                  mu*(4 u_t(h) - u_t(2h)) / (2h), for several h (the best one is reported,
                  which flatters the baseline). Interpolation has no pressure at all.

Usage:
    python scripts/report.py infer-hidden --models diag_case1_velonly diag_case1_velonly_nut \\
        diag_case1_full_nut --systolic 1778 --diastolic 2400
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from .metrics import predict_wss_physical
from .predict import load_trained, predict_physical
from ..config import METRICS_DIR, MODELS_DIR
from ..data.full_export import SPLIT_TEST, SPLIT_TRAIN, load_full_snapshot
from ..data.registry import cases_by_id, load_registry

MU = 0.0035
H_MM = (0.25, 0.5, 1.0, 1.5)


def wss_scores(Wp, snap):
    Wt, out = snap["wall_wss"], {}
    for tag, m in (("wall", np.ones(len(Wt), bool)), ("anz", snap["wall_aneurysm"])):
        mp, mt = np.linalg.norm(Wp[m], axis=1), np.linalg.norm(Wt[m], axis=1)
        cos = np.sum(Wp[m] * Wt[m], 1) / np.maximum(mp * mt, 1e-12)
        out[f"{tag}_vec_rel_l2"] = float(np.linalg.norm(Wp[m] - Wt[m]) / np.linalg.norm(Wt[m]))
        out[f"{tag}_mag_rel_l2"] = float(np.linalg.norm(mp - mt) / np.linalg.norm(mt))
        out[f"{tag}_pearson_mag"] = float(np.corrcoef(mp, mt)[0, 1])
        out[f"{tag}_dir_cos_p50"] = float(np.median(cos))
        out[f"{tag}_q99_ratio"] = float(np.quantile(mp, 0.99) / np.quantile(mt, 0.99))
    return out


def pressure_scores(model, snap, rec, phase):
    """Gauge pressure (vs the outlet face) on test cubes and on the wall; the pressure drop."""
    X = snap["vol_xyz"]
    test = snap["vol_split"] == SPLIT_TEST
    kw = dict(diameter_cm=rec.inlet_diameter_cm, disease_flag=rec.disease_flag, phase=phase, beta=1.0)
    p_vol = predict_physical(model, X[test], **kw)["p"]
    p_wall = predict_physical(model, snap["wall_xyz"], **kw)["p"]
    p_in = predict_physical(model, snap["inlet_xyz"], **kw)["p"].mean()
    p_out = predict_physical(model, snap["outlet_xyz"], **kw)["p"].mean()
    po = snap["outlet_p"].mean()
    tv, tw = snap["vol_p"][test] - po, snap["wall_p"] - po
    dv, dw = (p_vol - p_out) - tv, (p_wall - p_out) - tw
    in_cfd = snap["vol_p"][cKDTree(X).query(snap["inlet_xyz"])[1]].mean()
    return {"p_vol_rel_l2": float(np.linalg.norm(dv) / np.linalg.norm(tv)),
            "p_wall_rel_l2": float(np.linalg.norm(dw) / np.linalg.norm(tw)),
            "p_wall_pearson": float(np.corrcoef(p_wall, snap["wall_p"])[0, 1]),
            "dp_pinn_pa": float(p_in - p_out), "dp_cfd_pa": float(in_cfd - po)}


def interp_wss(snap):
    X, T = snap["vol_xyz"], snap["vol_uvw"]
    tr = snap["vol_split"] == SPLIT_TRAIN
    W, n = snap["wall_xyz"], snap["wall_normals"]
    f = LinearNDInterpolator(np.vstack([X[tr], W]), np.vstack([T[tr], np.zeros((len(W), 3))]))
    tang = lambda U: U - np.sum(U * n, 1, keepdims=True) * n  # noqa: E731
    best = {}
    for h_mm in H_MM:
        h = h_mm / 1000
        u1, u2 = f(W + h * n), f(W + 2 * h * n)
        for order, est in (("1st", MU * tang(u1) / h), ("2nd", MU * (4 * tang(u1) - tang(u2)) / (2 * h))):
            ok = np.isfinite(est).all(1)
            est = np.where(ok[:, None], est, 0.0)
            s = wss_scores(est, snap)
            s.update(h_mm=h_mm, order=order, frac_undefined=float(1 - ok.mean()))
            key = s["wall_mag_rel_l2"]
            if not best or key < best["wall_mag_rel_l2"]:
                best = s
    return best


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--case", type=int, default=1)
    ap.add_argument("--systolic", type=int, default=1778)
    ap.add_argument("--diastolic", type=int, default=2400)
    ap.add_argument("--no-interp", action="store_true")
    ap.add_argument("--device", default="cpu")


def run(args: argparse.Namespace) -> None:
    rec = cases_by_id(load_registry())[args.case]
    rows = []
    for phase, t in (("systolic", args.systolic), ("diastolic", args.diastolic)):
        snap = load_full_snapshot(args.case, t)
        if not args.no_interp:
            rows.append({"field": "linear interpolation", "phase": phase, **interp_wss(snap)})
            print(rows[-1], flush=True)
        for m in args.models:
            ck = MODELS_DIR / m / "best_model.pt"
            if not ck.exists():
                print(f"skip {m}: no checkpoint")
                continue
            model = load_trained(ck, device=args.device)
            Wp = predict_wss_physical(model, snap["wall_xyz"], snap["wall_normals"], rec.inlet_diameter_cm,
                                      rec.disease_flag, phase, beta=1.0)["wss_components"]
            rows.append({"field": m, "phase": phase, **wss_scores(Wp, snap),
                         **pressure_scores(model, snap, rec, phase)})
            print(rows[-1], flush=True)
    out = METRICS_DIR / "_full"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"infer_hidden_case{args.case}.json").write_text(json.dumps(rows, indent=1, default=float))
