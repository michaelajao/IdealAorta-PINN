"""CFD-interpolation baseline for leave-one-diameter-out (Block 4 comparator).

Answers "what does naive interpolation of the CFD velocity fields across diameter
give on the held diameter?" -- the floor a diameter-conditioned surrogate must beat
to justify the word "surrogate". For each held case it takes the SAME-symmetry
training cases at the bracketing diameters, registers their CFD velocity onto the
held case's point cloud (nearest neighbour in per-cloud-standardized coordinates,
since the idealized meshes are scaled copies but not point-matched), blends them
linearly in diameter, and scores against the held CFD field.

Honesty constraints (Gap 10): the meshes are not point-matched, so the registration
is a nearest-neighbour correspondence in a normalized frame, NOT a true field
transfer; velocities are blended raw (no inlet-velocity rescaling). Crucially, this
baseline is *handed the held case's own mesh* (its point cloud) and only interpolates
the solution onto it, so it is an OPTIMISTIC floor -- the PINN sees neither the held
mesh nor its field. State all three in the caption. ``--mode nearest`` uses the
single nearest diameter (no blend) instead.

Output mirrors ``velocity_metrics`` schema into ``report/metrics/<out>/velocity.json``
so ``make_ablation_table.py`` can fold it into the comparator table (T7) as a row
whose "PINN" column is the baseline-under-test.

Usage:
    python scripts/baseline_cfd_interp.py --held 4 5 6 --mode blend --out cfd_interp_f2p3
    python scripts/baseline_cfd_interp.py --held 4 5 6 --mode nearest --out nn_diam_f2p3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idealaorta_pinn.analysis.metrics import _flow_pair, _rel_l2, write_report  # noqa: E402
from idealaorta_pinn.config import PROJECT_ROOT  # noqa: E402
from idealaorta_pinn.data.cache import load_points  # noqa: E402
from idealaorta_pinn.data.registry import cases_by_id, load_registry  # noqa: E402

PHASES = ("systolic", "diastolic")


def _xyz_uvw(rec, phase):
    df = load_points(rec, "3D", phase)
    if df is None or not {"u", "v", "w"}.issubset(df.columns):
        return None
    coords = df[["x", "y", "z"]].to_numpy(float)
    vel = df[["u", "v", "w"]].to_numpy(float)
    if float(np.abs(vel).max()) < 1e-8:           # degenerate (no resolved flow)
        return None
    return coords, vel


def _standardize(coords: np.ndarray) -> np.ndarray:
    """Per-cloud centre + isotropic half-extent scaling -> ~[-1,1]^3.

    Brings two diameter-scaled idealized geometries into a common frame so a
    nearest-neighbour query maps roughly corresponding anatomical locations.
    """
    c = coords - coords.mean(axis=0)
    half = 0.5 * float((coords.max(axis=0) - coords.min(axis=0)).max())
    return c / max(half, 1e-9)


def _register_predict(src_coords, src_vel, dst_coords) -> np.ndarray:
    """Nearest-neighbour transfer of src velocity onto dst points (normalized frame)."""
    tree = cKDTree(_standardize(src_coords))
    _, idx = tree.query(_standardize(dst_coords), k=1)
    return src_vel[idx]


def _neighbors(by, held_rec, mode):
    """Return [(rec, weight), ...] same-symmetry training neighbours for one held case."""
    d_h = held_rec.inlet_diameter_cm
    cand = [r for r in by.values()
            if r.symmetry == held_rec.symmetry and r.disease_flag == held_rec.disease_flag
            and abs(r.inlet_diameter_cm - d_h) > 1e-6]
    if not cand:
        return [], "none"
    if mode == "nearest":
        best = min(cand, key=lambda r: abs(r.inlet_diameter_cm - d_h))
        return [(best, 1.0)], f"nearest({best.inlet_diameter_cm}cm)"
    lo = max((r for r in cand if r.inlet_diameter_cm < d_h),
             key=lambda r: r.inlet_diameter_cm, default=None)
    hi = min((r for r in cand if r.inlet_diameter_cm > d_h),
             key=lambda r: r.inlet_diameter_cm, default=None)
    if lo is not None and hi is not None:
        d_lo, d_hi = lo.inlet_diameter_cm, hi.inlet_diameter_cm
        w_lo = (d_hi - d_h) / (d_hi - d_lo)
        return [(lo, w_lo), (hi, 1.0 - w_lo)], f"interp({d_lo}+{d_hi}cm)"
    one = lo or hi                                  # extrapolation: single bracketing side
    return [(one, 1.0)], f"extrap({one.inlet_diameter_cm}cm)"


def main() -> None:
    ap = argparse.ArgumentParser(description="CFD-interpolation baseline (Block 4).")
    ap.add_argument("--held", type=int, nargs="+", required=True, help="Held-out case ids.")
    ap.add_argument("--mode", choices=["blend", "nearest"], default="blend")
    ap.add_argument("--phases", nargs="*", default=list(PHASES))
    ap.add_argument("--out", type=str, default="cfd_interp")
    args = ap.parse_args()

    by = cases_by_id(load_registry())
    rows = []
    for cid in args.held:
        held = by[cid]
        neigh, tag = _neighbors(by, held, args.mode)
        if not neigh:
            print(f"[cfd-interp] case {cid}: no same-symmetry neighbours; skipped")
            continue
        for phase in args.phases:
            tgt = _xyz_uvw(held, phase)
            if tgt is None:
                print(f"[cfd-interp] case {cid} {phase}: degenerate/absent held field; skipped")
                continue
            coords_h, vel_h = tgt
            pred = np.zeros_like(vel_h)
            ok = True
            for rec, w in neigh:
                src = _xyz_uvw(rec, phase)
                if src is None:
                    ok = False
                    break
                pred += w * _register_predict(src[0], src[1], coords_h)
            if not ok:
                print(f"[cfd-interp] case {cid} {phase}: a neighbour field missing; skipped")
                continue

            true_speed = np.linalg.norm(vel_h, axis=1)
            phase_u_ref = max(float(np.quantile(true_speed, 0.995)), 1e-6)
            rows.append({
                "case": cid, "phase": phase, "kind": "cfd_interp",
                "n_points": int(len(coords_h)),
                "mode": args.mode, "neighbors": tag,
                "vel_nrmse_phase": float(np.sqrt(np.mean((pred - vel_h) ** 2)) / phase_u_ref),
                "vel_rel_l2": _rel_l2(pred, vel_h),
                "diameter_cm": held.inlet_diameter_cm,
                **_flow_pair(vel_h, pred, phase_u_ref),
            })
            print(f"[cfd-interp] case {cid:>2} {phase:<9} {tag:<16} "
                  f"NRMSE={rows[-1]['vel_nrmse_phase']:.3f}  relL2={rows[-1]['vel_rel_l2']:.3f}  "
                  f"recirc CFD/base={rows[-1]['recirc_cfd']:.3f}/{rows[-1]['recirc_pinn']:.3f}")

    if not rows:
        print("[cfd-interp] no rows produced.")
        return
    from idealaorta_pinn.config import METRICS_DIR
    out_dir = METRICS_DIR / args.out
    write_report(rows, "velocity", f"{args.out}: CFD-interpolation baseline ({args.mode})",
                 out_dir=out_dir)
    print(f"[cfd-interp] wrote {(out_dir / 'velocity.json').relative_to(PROJECT_ROOT)} "
          f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
