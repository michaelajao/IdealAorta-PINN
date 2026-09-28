"""Sparse-reconstruction study CLI: reference audits, comparators, neural fields, analyses.

The logic lives in ``idealaorta_pinn.reconstruction``; this file is only the
command-line front end. Studies are defined in ``configs/reconstruction/*.yaml``;
results go to ``report/metrics/reconstruction/`` and checkpoints to ``models/``.

Subcommands
-----------
audit        Momentum budget and pressure oracle of the CFD fields at one instant
             (du/dt from the nearest snapshots within 50 ms unless given).
wss-oracle   Wall-shear estimators from the complete (or subsampled) CFD velocity.
baseline     One comparator (linear, rbf, rbf_tuned, rbf_spacetime) or the ``cfd``
             pipeline floor on one problem.
train        Fit and score one neural-field arm (``--profile N`` times N steps only).
diagnose     Residual-by-wall-distance and pipeline pressure of a trained arm.
jobs         Expand a study into command lines for scripts/queue_jobs.sh.
select       Physics weight per arm from held-out observation error (pilot_select).
summarize    Every method per case / window / grid for a study.
confirm      Pre-registered hypothesis tests of a confirmatory study.

Usage:
    python scripts/reconstruct.py audit --case 1 --t 1780
    python scripts/reconstruct.py baseline --method rbf --case 4 --times 1775 1780 --target 1780 --grid 2.5 --seed 0
    python scripts/reconstruct.py train --arm cont --case 4 --times 1775 1780 --target 1780 --wphys 0.01
    python scripts/reconstruct.py jobs confirm --kind train > jobs.txt && scripts/queue_jobs.sh jobs.txt 6 gpu
    python scripts/reconstruct.py confirm confirm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idealaorta_pinn.config import RECON_METRICS_DIR  # noqa: E402


def _write(sub: str, stem: str, payload: dict) -> Path:
    out = RECON_METRICS_DIR / sub
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stem}.json"
    path.write_text(json.dumps(payload, indent=1, default=float))
    return path


def _problem_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--case", type=int, required=True)
    p.add_argument("--times", type=int, nargs=2, required=True, help="window snapshots (ms)")
    p.add_argument("--target", type=int, required=True, help="scored snapshot (ms)")
    p.add_argument("--grid", type=float, default=2.5, help="observation grid spacing (mm)")
    p.add_argument("--seed", type=int, default=0, help="observation-grid draw")
    p.add_argument("--min-wall-mm", type=float, default=0.0,
                   help="drop observations nearer the wall than this (mm)")


def cmd_audit(args) -> None:
    from idealaorta_pinn.reconstruction.reference import audit
    from idealaorta_pinn.reconstruction.study import snapshot_neighbours
    before, after = snapshot_neighbours(args.case, args.t) if args.before is None and args.after is None \
        else (args.before, args.after)
    res = audit(args.case, args.t, before, after, with_pressure=not args.no_pressure, forms=args.forms)
    path = _write("reference", f"case{args.case:02d}_t{args.t}", res)
    bulk = res["budget"]["bulk(>1.5mm)"]["residual_median_over_conv_median"]
    print(f"[audit] case {args.case} t {args.t} ms ({res['dudt']}): median residual / convection "
          + ", ".join(f"{k} {v:.3f}" for k, v in bulk.items()))
    if "pressure_oracle" in res:
        po = res["pressure_oracle"]
        print("[audit] pressure rel L2: " + ", ".join(f"{k} {v['p_rel_l2_sub']:.4f}" for k, v in po.items()
                                                     if isinstance(v, dict)))
    print(f"[audit] wrote {path}")


def cmd_wss_oracle(args) -> None:
    from idealaorta_pinn.reconstruction.reference import wss_oracle
    res = wss_oracle(args.case, args.t, sources=args.sources, frac=args.frac, seed=args.seed)
    path = _write("wss_oracle", f"case{args.case:02d}_t{args.t}", res)
    for r in res["rows"]:
        if r["estimator"] == "newton2" and "aneurysm" in r:
            print(f"[wss] {r['source']} h={r['h_mm']} mm: aneurysm |WSS| {r['aneurysm']['mag_rel_l2']:.3f}, "
                  f"whole wall {r['wall']['mag_rel_l2']:.3f}")
    print(f"[wss] wrote {path}")


def cmd_baseline(args) -> None:
    from idealaorta_pinn.reconstruction.interpolation import run_comparator
    r = run_comparator(args.method, args.case, args.times, args.target, args.grid, args.seed,
                       min_wall_mm=args.min_wall_mm)
    print(f"[baseline] {r['run_name']}: velocity {r['velocity']['vel_rel_l2']:.4f}, "
          f"aneurysm WSS {r.get('wss', {}).get('wss_mag_rel_l2', float('nan')):.4f}, "
          f"pressure {r['pressure']['unsteady+mut']['p_rel_l2']:.4f} ({r['seconds']} s)")


def cmd_train(args) -> None:
    from idealaorta_pinn.reconstruction.training import ArmSpec, train_arm
    spec = ArmSpec(arm=args.arm, closure=args.closure, case=args.case, times=args.times, target=args.target,
                   grid=args.grid, seed=args.seed, init_seed=args.init_seed, wphys=args.wphys,
                   steps=args.steps, lr=args.lr, min_wall_mm=args.min_wall_mm)
    r = train_arm(spec, device=args.device, profile_steps=args.profile)
    if args.profile:
        print(json.dumps(r))
        return
    print(f"[train] {r['run_name']}: velocity {r['velocity']['vel_rel_l2']:.4f}, "
          f"aneurysm WSS {r.get('wss', {}).get('wss_mag_rel_l2', float('nan')):.4f}, "
          f"pressure {r['pressure']['unsteady+mut']['p_rel_l2']:.4f} ({r['seconds']} s)")


def cmd_diagnose(args) -> None:
    from idealaorta_pinn.reconstruction.training import diagnose
    for name in args.names:
        out = diagnose(name, device=args.device)
        print(json.dumps(out, indent=1, default=float))


def cmd_jobs(args) -> None:
    from idealaorta_pinn.reconstruction.study import jobs, load_study
    print("\n".join(jobs(load_study(args.study), args.kind)))


def cmd_select(args) -> None:
    from idealaorta_pinn.reconstruction.study import load_study, select_weights
    out = select_weights(load_study(args.study))
    for row in out["table"]:
        print(f"{row['arm']:9s} {row['closure']:7s} w={row['wphys']:<5g} val_obs {row['val_obs_rel_l2']:.4f}")
    print("selected:", out["selected"])


def cmd_summarize(args) -> None:
    from idealaorta_pinn.reconstruction.study import load_study, summarize
    out = summarize(load_study(args.study))
    for unit, methods in out.items():
        print(f"\n{unit}")
        for m, v in methods.items():
            mean = lambda k: sum(v[k]) / len(v[k]) if v.get(k) else float("nan")  # noqa: E731
            print(f"  {m:24s} n={len(v['vel'])} vel {mean('vel'):.3f}  wss {mean('wss'):.3f}  "
                  f"p(unsteady+mut) {mean('p_unsteady+mut'):.3f}  p(own head) {mean('p_own'):.3f}")


def cmd_confirm(args) -> None:
    from idealaorta_pinn.reconstruction.study import confirm, load_study
    study = load_study(args.study)
    out = confirm(study)
    cand, comp = study["hypotheses"]["candidate"], study["hypotheses"]["comparator"]
    print(f"{'case':>4} {'t':>5} | pressure: {cand:>6} {comp:>6} linear   cfd | "
          f"aneurysm WSS: {cand:>6} {comp:>6} | velocity: {cand:>6} {comp:>6}")
    for u in out["units"]:
        a, b = u[cand], u[comp]
        print(f"{u['case']:4d} {u['target_ms']:5d} | {a['p']:16.3f} {b['p']:6.3f} {u['linear']['p']:6.3f} "
              f"{u['cfd']['p']:5.3f} | {a['wss']:20.3f} {b['wss']:6.3f} | {a['vel']:16.3f} {b['vel']:6.3f}")
    for h in ("H1_pressure", "H2_wss"):
        print(h, out[h])
    print("guardrail_velocity:", out["guardrail_velocity"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Sparse-reconstruction study.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("audit", help="momentum budget + pressure oracle of the CFD fields")
    p.add_argument("--case", type=int, required=True)
    p.add_argument("--t", type=int, required=True, help="snapshot (ms)")
    p.add_argument("--before", type=int, default=None, help="earlier snapshot for du/dt (ms)")
    p.add_argument("--after", type=int, default=None, help="later snapshot for du/dt (ms)")
    p.add_argument("--no-pressure", action="store_true")
    p.add_argument("--forms", nargs="*", default=None, help="subset of momentum forms for the pressure oracle")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("wss-oracle", help="wall-shear estimators from CFD velocity")
    p.add_argument("--case", type=int, required=True)
    p.add_argument("--t", type=int, required=True)
    p.add_argument("--sources", nargs="+", default=["full"], choices=["full", "train", "frac"])
    p.add_argument("--frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_wss_oracle)

    p = sub.add_parser("baseline", help="one comparator on one problem")
    p.add_argument("--method", required=True, choices=["cfd", "linear", "rbf", "rbf_tuned", "rbf_spacetime"])
    _problem_args(p)
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("train", help="fit and score one neural-field arm")
    p.add_argument("--arm", required=True, choices=["data", "cont", "steady", "unsteady"])
    p.add_argument("--closure", default="oracle", choices=["oracle", "lam"])
    _problem_args(p)
    p.add_argument("--init-seed", type=int, default=0, help="network / optimizer seed")
    p.add_argument("--wphys", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--profile", type=int, default=0, help="time N steps and exit")
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("diagnose", help="residual and pipeline diagnostics of trained arms")
    p.add_argument("names", nargs="+", help="run names under models/")
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("jobs", help="expand a study into job lines")
    p.add_argument("study", help="name under configs/reconstruction/ or a YAML path")
    p.add_argument("--kind", required=True, choices=["train", "baseline", "robustness", "reference"])
    p.set_defaults(func=cmd_jobs)

    for name, func, help_ in (("select", cmd_select, "physics-weight selection table"),
                              ("summarize", cmd_summarize, "per-unit table of every method")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("study")
        p.set_defaults(func=func)

    p = sub.add_parser("confirm", help="pre-registered hypothesis tests")
    p.add_argument("study")
    p.set_defaults(func=cmd_confirm)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
