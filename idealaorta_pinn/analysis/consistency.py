"""Does the PINN beat plain interpolation where the revision claims it should?

Two checks on the whole-vessel exports, both on CPU:

``--insample``  the in-sample comparator. The Case 1 full-export runs learn from the even
    cubes of the checkerboard; here the same even-cube nodes (plus the zero-velocity wall)
    are interpolated to the test cubes by nearest neighbour and by linear (Delaunay)
    interpolation, and scored exactly like the model.

default (LODO)  physical consistency on held-out cases, for the model, the three
    CFD-interpolation baselines of ``report.py score-full`` and the CFD itself:
      div_rel   rms(div u) / rms(|grad u|) from least-squares gradients on the fluid
                nodes (> 1 mm from the wall), the same estimator for every field
      no-slip   rms |u| on the held case's wall over the phase's 99.5th-pct speed
      pressure  rel-L2 of p - p_outlet on test cubes, and the inlet-to-outlet drop
                ratio (predicted / CFD)

Usage:
    python scripts/report.py consistency --insample --model diag_case1_full_nut --cases 1 \\
        --systolic 1778 --diastolic 2400
    python scripts/report.py consistency --model lodo_full_hold2p6 --cases 7 8 9
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from .metrics import BASELINE_MODES, _baseline_neighbours, field_error_stats
from .predict import load_trained, predict_physical
from ..config import METRICS_DIR, MODELS_DIR
from ..data.full_export import SPLIT_TEST, SPLIT_TRAIN, load_full_snapshot
from ..data.registry import cases_by_id, load_registry


def _beta(rec):
    return float(rec.beta) if rec.beta is not None else 1.0


def _frame(snap):
    X = snap["vol_xyz"]
    return X.mean(axis=0), max(0.5 * float((X.max(axis=0) - X.min(axis=0)).max()), 1e-9)


def div_rel(X, U, centres, k=17):
    """rms(div u) / rms(|grad u|_F) at ``centres`` from weighted least-squares gradients."""
    tree = cKDTree(X)
    _, nb = tree.query(X[centres], k=k)
    nb = nb[:, 1:]
    dX = X[nb] - X[centres, None, :]
    dU = U[nb] - U[centres, None, :]
    w = 1.0 / np.maximum(np.einsum("nkd,nkd->nk", dX, dX), 1e-16)
    A = dX * w[..., None]
    G = np.linalg.solve(np.einsum("nki,nkj->nij", A, dX), np.einsum("nki,nkc->nic", A, dU))
    div = np.einsum("nii->n", G)
    return float(np.sqrt(np.mean(div ** 2)) / np.sqrt(np.mean(np.einsum("nic,nic->n", G, G))))


def _centres(snap, n=60_000, seed=0):
    d, _ = cKDTree(snap["wall_xyz"]).query(snap["vol_xyz"])
    idx = np.flatnonzero(d > 1e-3)
    return np.random.default_rng(seed).choice(idx, size=min(n, len(idx)), replace=False)


def _vel_scores(P, T, q, sac):
    s = field_error_stats(P, T, q)
    out = {"vel_nrmse_vec": s["vel_nrmse_vec"], "frac_err_gt_25": s["frac_err_gt_25"]}
    for tag, m in (("sac", sac), ("parent", ~sac)):
        out[f"{tag}_nrmse"] = field_error_stats(P[m], T[m], q)["vel_nrmse_vec"]
    return out


def _pressure_scores(Pp, snap, test, p_in_pred, p_out_pred):
    """Gauge pressure (relative to the outlet face) on test cubes, and the pressure drop."""
    p_out = snap["outlet_p"].mean()
    t = snap["vol_p"][test] - p_out
    e = (Pp - p_out_pred) - t
    drop_cfd = float(snap["vol_p"][_near(snap, "inlet")].mean() - p_out)
    return {"p_rel_l2": float(np.linalg.norm(e) / max(np.linalg.norm(t), 1e-12)),
            "dp_ratio": float((p_in_pred - p_out_pred) / drop_cfd) if abs(drop_cfd) > 1e-9 else np.nan}


def _near(snap, face, tol=1e-3):
    """Volume nodes within ``tol`` of the inlet or outlet face, inside that face's disk
    (the upper limb also crosses x = 0, so the axial position alone is not enough)."""
    F = snap[f"{face}_xyz"]
    c, r = F.mean(axis=0), float(np.hypot(F[:, 1] - F[:, 1].mean(), F[:, 2] - F[:, 2].mean()).max())
    X = snap["vol_xyz"]
    return (np.abs(X[:, 0] - c[0]) < tol) & (np.hypot(X[:, 1] - c[1], X[:, 2] - c[2]) <= r + tol)


def insample(model_name, rec, times, device):
    model = load_trained(MODELS_DIR / model_name / "best_model.pt", device=device)
    rows = []
    for phase, t_ms in times.items():
        s = load_full_snapshot(rec.case_id, t_ms)
        X, T = s["vol_xyz"], s["vol_uvw"]
        test, tr = s["vol_split"] == SPLIT_TEST, s["vol_split"] == SPLIT_TRAIN
        q = float(np.quantile(np.linalg.norm(T, axis=1), 0.995))
        src = np.vstack([X[tr], s["wall_xyz"]])                    # training nodes + no-slip wall
        srcU = np.vstack([T[tr], np.zeros((len(s["wall_xyz"]), 3))])
        _, i = cKDTree(src).query(X[test])
        nn = srcU[i]
        lin = LinearNDInterpolator(src, srcU)(X[test])
        bad = ~np.isfinite(lin).all(axis=1)
        lin[bad] = nn[bad]
        p = predict_physical(model, X[test], rec.inlet_diameter_cm, rec.disease_flag, phase, beta=_beta(rec))
        pinn = np.c_[p["u"], p["v"], p["w"]]
        sac = s["vol_sac"][test]
        for name, P in (("PINN", pinn), ("nearest-node", nn), ("linear", lin)):
            rows.append({"case": rec.case_id, "phase": phase, "field": name,
                         **_vel_scores(P, T[test], q, sac)})
            print(rows[-1], flush=True)
    return rows


def lodo(model_name, recs, by, times, device):
    ck = MODELS_DIR / model_name / "best_model.pt"
    model = load_trained(ck, device=device)
    train = torch.load(ck, map_location="cpu", weights_only=False)["config"]["data"]["train_cases"]
    rows = []
    for rec in recs:
        for phase, t_ms in times.items():
            start = len(rows)
            s = load_full_snapshot(rec.case_id, t_ms)
            X, T = s["vol_xyz"], s["vol_uvw"]
            test = s["vol_split"] == SPLIT_TEST
            q = float(np.quantile(np.linalg.norm(T, axis=1), 0.995))
            cen = _centres(s)
            p_out_cfd = s["outlet_p"].mean()
            inl, outl = _near(s, "inlet"), _near(s, "outlet")

            def row(name, U_all, P_all, U_wall):
                return {"case": rec.case_id, "phase": phase, "field": name,
                        "div_rel": div_rel(X, U_all, cen),
                        "noslip_rms": float(np.sqrt(np.mean(np.sum(U_wall ** 2, axis=1))) / q),
                        **({} if P_all is None else _pressure_scores(
                            P_all[test], s, test, P_all[inl].mean(), P_all[outl].mean())),
                        "vel_nrmse_vec": field_error_stats(U_all[test], T[test], q)["vel_nrmse_vec"]}

            rows.append(row("CFD", T, s["vol_p"], np.zeros((len(s["wall_xyz"]), 3))))
            pa = predict_physical(model, X, rec.inlet_diameter_cm, rec.disease_flag, phase, beta=_beta(rec))
            pw = predict_physical(model, s["wall_xyz"], rec.inlet_diameter_cm, rec.disease_flag, phase,
                                  beta=_beta(rec))
            rows.append(row("PINN", np.c_[pa["u"], pa["v"], pa["w"]], pa["p"],
                            np.c_[pw["u"], pw["v"], pw["w"]]))
            ch, hh = _frame(s)
            for mode in BASELINE_MODES:
                neigh, tag = _baseline_neighbours(by, rec, train, mode)
                if not neigh:
                    continue
                U = Uw = Pb = 0.0
                for src_rec, w in neigh:
                    ss = load_full_snapshot(src_rec.case_id, t_ms)
                    cs, hs = _frame(ss)
                    # the source's fluid nodes and its wall (zero velocity, wall pressure)
                    SX = np.vstack([ss["vol_xyz"], ss["wall_xyz"]])
                    SU = np.vstack([ss["vol_uvw"], np.zeros((len(ss["wall_xyz"]), 3))])
                    SP = np.concatenate([ss["vol_p"], ss["wall_p"]]) - ss["outlet_p"].mean()
                    tree = cKDTree((SX - cs) / hs)
                    _, iv = tree.query((X - ch) / hh)
                    _, iw = tree.query((s["wall_xyz"] - ch) / hh)
                    u, uw, pg = SU[iv], SU[iw], SP[iv]
                    if mode == "nearest_scaled":            # Q fixed: U ~ 1/d^2, dynamic p ~ 1/d^4
                        r = src_rec.inlet_diameter_cm / rec.inlet_diameter_cm
                        u, uw, pg = u * r ** 2, uw * r ** 2, pg * r ** 4
                    U, Uw, Pb = U + w * u, Uw + w * uw, Pb + w * pg
                rows.append(row(f"baseline:{mode}", U, Pb + p_out_cfd, Uw))
            for r in rows[start:]:
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    return rows


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", required=True)
    ap.add_argument("--cases", type=int, nargs="+", required=True)
    ap.add_argument("--systolic", type=int, default=1780)
    ap.add_argument("--diastolic", type=int, default=2400)
    ap.add_argument("--insample", action="store_true")
    ap.add_argument("--device", default="cpu")


def run(args: argparse.Namespace) -> None:
    by = cases_by_id(load_registry())
    times = {"systolic": args.systolic, "diastolic": args.diastolic}
    if args.insample:
        rows = [r for c in args.cases for r in insample(args.model, by[c], times, args.device)]
    else:
        rows = lodo(args.model, [by[c] for c in args.cases], by, times, args.device)
    out = METRICS_DIR / "_full"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"consistency_{'insample_' if args.insample else ''}{args.model}"
    (out / f"{tag}.json").write_text(json.dumps(rows, indent=1, default=float))
    print(f"wrote {out / (tag + '.json')}")
