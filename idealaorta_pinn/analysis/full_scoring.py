"""Score trained models, and the CFD-interpolation baselines, against the whole-vessel exports.

Every model is scored on the same points, whatever it was trained on:

    test cubes   the withheld odd cubes of the 3D checkerboard (minus the monitoring
                 fifth), split into the sac (nodes nearest the aneurysm wall zone) and
                 the parent vessel
    slabs        test-cube nodes within 0.5 mm of the XY (z = 0) and XZ (y = 3 cm)
                 planes, available at any exported instant
    planes       the older XY/XZ streamline exports, only when the scored instant is
                 the one they were exported at
    wall         wall shear stress on the whole wall (normals oriented by the volume)
                 and on the aneurysm wall zone alone, as vector and magnitude rel-L2
    nu_t         the model's eddy viscosity against the CFD's mu_t / rho on test cubes

With ``--baselines`` each scored case also gets the three CFD-interpolation baselines
of ``report.py rescore`` (blend, nearest, Q/A-scaled nearest of the same-symmetry training
cases, each case mapped to its own centre-and-extent frame), built from the training
cases' full volume and wall at the same instant and scored on the very same nodes.

Velocity errors use ``field_error_stats`` (vector error over the phase's 99.5th-
percentile CFD speed), as in ``report.py rescore``.

Usage:
    python scripts/report.py score-full --models stageA_case1_s12 diag_case1_full_nut --cases 1
    python scripts/report.py score-full --models lodo_full_hold2p3 --cases 4 5 6 \\
        --systolic 1780 --diastolic 2400 --baselines --tag lodo_full_hold2p3
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from scipy.spatial import cKDTree

from .metrics import (BASELINE_MODES, _baseline_neighbours,
                                              field_error_stats, predict_wss_physical,
                                              predicted_divergence, velocity_metrics)
from .predict import load_trained, predict_physical
from ..config import METRICS_DIR, MODELS_DIR, TABLES_DIR
from ..data.full_export import SPLIT_TEST, load_full_snapshot
from ..data.registry import cases_by_id, load_registry

KEYS = ("vel_nrmse_vec", "err_p95", "frac_err_gt_25", "nrmse_vec_slow", "overshoot_q995")
SLABS = {"XY": (2, 0.0), "XZ": (1, 0.03)}      # plane: (normal axis, position in m)
SLAB_HALF = 0.5e-3


def _stats(pred, true, q):
    s = field_error_stats(pred, true, q)
    return {k: s[k] for k in KEYS}


def _beta(rec):
    return float(rec.beta) if rec.beta is not None else 1.0


def _uvw(model, X, rec, phase):
    p = predict_physical(model, X, rec.inlet_diameter_cm, rec.disease_flag, phase, beta=_beta(rec))
    return np.c_[p["u"], p["v"], p["w"]]


@torch.no_grad()
def _nut_physical(model, X, rec, phase):
    """Model eddy viscosity mu_t (Pa s) at physical points."""
    n = model.normalizer
    mu = np.array([n.diameter_nd(rec.inlet_diameter_cm), _beta(rec), float(rec.disease_flag),
                   1.0 if phase == "systolic" else 0.0], dtype=np.float32)
    cs = n.coords_std(X).astype(np.float32)
    out = []
    for i in range(0, len(cs), 100_000):
        c = cs[i:i + 100_000]
        net_in = torch.tensor(np.hstack([c, np.tile(mu, (len(c), 1))]), device=model.device)
        out.append(model.networks["nut"](net_in).cpu().numpy().ravel())
    return np.concatenate(out) * n.U_ref * n.L * n.rho


def _velocity_block(P, snap, test, q):
    """Velocity scores of predictions ``P`` at the test-cube nodes of ``snap``."""
    X, T = snap["vol_xyz"][test], snap["vol_uvw"][test]
    row = {f"cubes_{k}": v for k, v in _stats(P, T, q).items()}
    sac = snap["vol_sac"][test]
    for tag, m in (("sac", sac), ("parent", ~sac)):
        if m.any():
            row[f"cubes_{tag}_nrmse"] = _stats(P[m], T[m], q)["vel_nrmse_vec"]
    for kind, (ax, pos) in SLABS.items():
        m = np.abs(X[:, ax] - pos) < SLAB_HALF
        if m.sum() > 50:
            row[f"{kind}slab_nrmse"] = _stats(P[m], T[m], q)["vel_nrmse_vec"]
    return row


def _wall_block(Wp, snap):
    """WSS scores of predicted wall-shear vectors ``Wp`` on the whole wall of ``snap``."""
    Wt, row = snap["wall_wss"], {}
    for tag, m in (("wall", np.ones(len(Wt), bool)), ("anz", snap["wall_aneurysm"])):
        if m.any():
            row[f"wss_{tag}_rel_l2"] = float(np.linalg.norm(Wp[m] - Wt[m]) / max(np.linalg.norm(Wt[m]), 1e-12))
            mp, mt = np.linalg.norm(Wp[m], axis=1), np.linalg.norm(Wt[m], axis=1)
            row[f"wss_{tag}_mag_rel_l2"] = float(np.linalg.norm(mp - mt) / max(np.linalg.norm(mt), 1e-12))
            row[f"wss_{tag}_q99_ratio"] = float(np.quantile(mp, 0.99) / np.quantile(mt, 0.99))
    return row


def score(model_name, ckpt, rec, times, block_mm, val_every, device, records):
    model = load_trained(MODELS_DIR / model_name / ckpt, device=device)
    rows = []
    for phase, t_ms in times.items():
        snap = load_full_snapshot(rec.case_id, t_ms, block_mm, val_every)
        test = snap["vol_split"] == SPLIT_TEST
        q = float(np.quantile(np.linalg.norm(snap["vol_uvw"], axis=1), 0.995))
        X = snap["vol_xyz"]
        row = {"model": model_name, "ckpt": ckpt, "case": rec.case_id, "phase": phase, "t_ms": t_ms,
               "n_test": int(test.sum())}
        row.update(_velocity_block(_uvw(model, X[test], rec, phase), snap, test, q))
        # the older plane exports, only at the instant they were exported at
        for kind in ("XY", "XZ"):
            t_plane = rec.files.get(kind, {}).get(phase, {}).get("time_s")
            if t_plane is not None and int(round(1000 * t_plane)) == t_ms:
                vm = velocity_metrics(model, records, rec.case_id, phase, kind=kind)
                if vm:
                    for k in KEYS + ("div_rel",):
                        row[f"{kind}_{k}"] = vm.get(k)
        sel = np.flatnonzero(test)[:: max(1, int(test.sum()) // 40_000)]
        row["cubes_div_rel"] = predicted_divergence(model, X[sel], rec.inlet_diameter_cm,
                                                    rec.disease_flag, phase, beta=_beta(rec))["div_rel"]
        Wp = predict_wss_physical(model, snap["wall_xyz"], snap["wall_normals"], rec.inlet_diameter_cm,
                                  rec.disease_flag, phase, beta=_beta(rec))["wss_components"]
        row.update(_wall_block(Wp, snap))
        mut_p = _nut_physical(model, X[test], rec, phase)
        mut_c = snap["vol_nut"][test] * model.normalizer.rho
        ok = mut_c > 1e-6
        row["mut_ratio_median"] = float(np.median(mut_p[ok] / mut_c[ok]))
        row["mut_log10_rmse"] = float(np.sqrt(np.mean(np.log10(mut_p[ok] / mut_c[ok]) ** 2)))
        rows.append(row)
        _echo(row)
    return rows


def _frame(snap):
    """Centre and half-extent of a case's fluid volume (the baselines' shared frame)."""
    X = snap["vol_xyz"]
    return X.mean(axis=0), max(0.5 * float((X.max(axis=0) - X.min(axis=0)).max()), 1e-9)


def baseline_rows(rec, by, train_cases, times, block_mm, val_every):
    """The three interpolation baselines for a held-out case, on the same nodes as the models."""
    rows = []
    for phase, t_ms in times.items():
        snap = load_full_snapshot(rec.case_id, t_ms, block_mm, val_every)
        test = snap["vol_split"] == SPLIT_TEST
        q = float(np.quantile(np.linalg.norm(snap["vol_uvw"], axis=1), 0.995))
        ch, hh = _frame(snap)
        dst_v = (snap["vol_xyz"][test] - ch) / hh
        dst_w = (snap["wall_xyz"] - ch) / hh
        for mode in BASELINE_MODES:
            neigh, tag = _baseline_neighbours(by, rec, train_cases, mode)
            if not neigh:
                continue
            P = Wp = 0.0
            for src_rec, w in neigh:
                s = load_full_snapshot(src_rec.case_id, t_ms, block_mm, val_every)
                cs, hs = _frame(s)
                _, iv = cKDTree((s["vol_xyz"] - cs) / hs).query(dst_v)
                _, iw = cKDTree((s["wall_xyz"] - cs) / hs).query(dst_w)
                vel, wss = s["vol_uvw"][iv], s["wall_wss"][iw]
                if mode == "nearest_scaled":              # Q fixed, U ~ 1/d^2, WSS ~ 1/d^3
                    r = src_rec.inlet_diameter_cm / rec.inlet_diameter_cm
                    vel, wss = vel * r ** 2, wss * r ** 3
                P, Wp = P + w * vel, Wp + w * wss
            row = {"model": f"baseline:{mode}", "ckpt": tag, "case": rec.case_id, "phase": phase,
                   "t_ms": t_ms, "n_test": int(test.sum())}
            row.update(_velocity_block(P, snap, test, q))
            row.update(_wall_block(Wp, snap))
            rows.append(row)
            _echo(row)
    return rows


def _echo(r):
    f = lambda k: r.get(k, float("nan"))  # noqa: E731
    print(f"{r['model']:26s} c{r['case']} {r['phase']:9s} cubes {f('cubes_vel_nrmse_vec'):.3f} "
          f"(sac {f('cubes_sac_nrmse'):.3f}, parent {f('cubes_parent_nrmse'):.3f})  "
          f">25% {f('cubes_frac_err_gt_25'):.3f}  |WSS| wall {f('wss_wall_mag_rel_l2'):.3f} "
          f"aneurysm {f('wss_anz_mag_rel_l2'):.3f}", flush=True)


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--cases", type=int, nargs="+", default=[1])
    ap.add_argument("--systolic", type=int, default=1778, help="systolic snapshot (ms)")
    ap.add_argument("--diastolic", type=int, default=2400, help="diastolic snapshot (ms)")
    ap.add_argument("--ckpt", nargs="+", default=["best_model.pt"])
    ap.add_argument("--baselines", action="store_true",
                    help="also score the interpolation baselines from the training cases")
    ap.add_argument("--train-cases", type=int, nargs="*", default=None,
                    help="baseline source cases (default: the first model's train_cases)")
    ap.add_argument("--block-mm", type=float, default=2.0)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default=None, help="output name (default: score_full_case<N>)")


def run(args: argparse.Namespace) -> None:

    records = load_registry()
    by = cases_by_id(records)
    times = {"systolic": args.systolic, "diastolic": args.diastolic}
    rows = []
    for cid in args.cases:
        for m in args.models:
            for ck in args.ckpt:
                if (MODELS_DIR / m / ck).exists():
                    rows += score(m, ck, by[cid], times, args.block_mm, args.val_every, args.device, records)
                else:
                    print(f"[score_full] skip {m}/{ck}: no checkpoint")
        if args.baselines:
            train = args.train_cases
            if train is None:
                ck = torch.load(MODELS_DIR / args.models[0] / args.ckpt[0], map_location="cpu",
                                weights_only=False)
                train = ck["config"]["data"]["train_cases"]
            rows += baseline_rows(by[cid], by, train, times, args.block_mm, args.val_every)
    tag = args.tag or f"score_full_case{'_'.join(map(str, args.cases))}"
    out = METRICS_DIR / "_full"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(rows, indent=1, default=float))

    cols = [("cubes_vel_nrmse_vec", "cubes"), ("cubes_sac_nrmse", "sac"), ("cubes_parent_nrmse", "parent"),
            ("cubes_frac_err_gt_25", ">25%"), ("cubes_overshoot_q995", "over"),
            ("XYslab_nrmse", "XY slab"), ("XZslab_nrmse", "XZ slab"),
            ("XY_vel_nrmse_vec", "XY"), ("XZ_vel_nrmse_vec", "XZ"),
            ("cubes_div_rel", "div_rel"), ("wss_wall_rel_l2", "WSS wall"), ("wss_wall_mag_rel_l2", "|WSS| wall"),
            ("wss_anz_rel_l2", "WSS aneurysm"), ("wss_anz_mag_rel_l2", "|WSS| aneurysm"),
            ("mut_ratio_median", "mu_t P/C")]
    md = [f"# Whole-vessel scores, cases {args.cases} ({args.systolic} / {args.diastolic} ms)", "",
          "| model | ckpt | case | phase | " + " | ".join(c[1] for c in cols) + " |",
          "|" + "---|" * (4 + len(cols))]
    for r in rows:
        md.append(f"| {r['model']} | {r['ckpt'].replace('_model.pt', '')} | {r['case']} | {r['phase']} | "
                  + " | ".join("-" if r.get(k) is None else f"{r[k]:.3f}" for k, _ in cols) + " |")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / f"{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
