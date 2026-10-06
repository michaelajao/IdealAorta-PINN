"""Command-line entry point of the reconstruction study.

Run from the repository root, ``python main.py <command> ...``. Studies are defined in
``configs/reconstruction/*.yaml``; outputs go to ``report/`` and checkpoints to
``models/`` (see ``idealaorta_pinn/config.py``).

Commands
--------
prepare      Split the raw whole-domain CFD exports into per-block parquet (once).
audit        Momentum budget and pressure check of the complete CFD fields at one instant.
wss-oracle   Wall-shear estimators applied to the complete CFD velocity.
baseline     One comparator (linear, rbf, rbf_tuned, rbf_spacetime) or the ``cfd``
             post-processing floor on one problem.
train        Fit and score one neural-field arm (``--profile N`` times N steps only).
diagnose     Residual-by-wall-distance and pipeline pressure of a trained momentum PINN.
jobs         Expand a study into command lines, one per run.
run-jobs     Run a file of command lines with bounded concurrency (GPU round-robin).
select       Physics weight per arm from the held-out observation error (dev_select).
summarize    Every method per case, window and grid for a study.
confirm      Prespecified hypothesis tests of a confirmatory study.
report       CSV tables and result figures of a confirmatory study.
figures      CFD-field figures (cfd) or reconstruction maps (maps), see idealaorta_pinn/plots.py.

Examples
--------
    python main.py prepare
    python main.py baseline --method rbf --case 4 --times 1775 1780 --target 1780
    python main.py train --arm cont --case 4 --times 1775 1780 --target 1780 --wphys 0.01
    python main.py jobs confirm --kind train > jobs_train.txt
    python main.py run-jobs jobs_train.txt --parallel 6 --gpus 0 1
    python main.py confirm confirm
    python main.py report confirm
    python main.py figures cfd maps
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from idealaorta_pinn.config import CHECKS_DIR, LOG_DIR


def _write_check(stem: str, payload: dict) -> Path:
    CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKS_DIR / f"{stem}.json"
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


# --------------------------------------------------------------------------- data
def cmd_prepare(args) -> None:
    from idealaorta_pinn.config import FULL_DIR, FULL_RAW_DIR
    from idealaorta_pinn.data import convert_all, raw_exports
    print(f"[prepare] {len(raw_exports())} raw exports under {FULL_RAW_DIR}")
    written = convert_all(cases=args.cases, overwrite=args.overwrite)
    print(f"[prepare] {len(written)} parquet files written to {FULL_DIR}")


# --------------------------------------------------------------------------- checks of the CFD fields
def cmd_audit(args) -> None:
    from idealaorta_pinn.reference import audit
    from idealaorta_pinn.study import snapshot_neighbours
    before, after = snapshot_neighbours(args.case, args.t) if args.before is None and args.after is None \
        else (args.before, args.after)
    res = audit(args.case, args.t, before, after, with_pressure=not args.no_pressure, forms=args.forms)
    path = _write_check(f"audit_case{args.case:02d}_t{args.t}", res)
    bulk = res["budget"]["bulk(>1.5mm)"]["residual_median_over_conv_median"]
    print(f"[audit] case {args.case} t {args.t} ms ({res['dudt']}): median residual / convection "
          + ", ".join(f"{k} {v:.3f}" for k, v in bulk.items()))
    if "pressure_oracle" in res:
        po = res["pressure_oracle"]
        print("[audit] pressure rel L2: " + ", ".join(f"{k} {v['p_rel_l2_sub']:.4f}" for k, v in po.items()
                                                     if isinstance(v, dict)))
    print(f"[audit] wrote {path}")


def cmd_wss_oracle(args) -> None:
    from idealaorta_pinn.reference import wss_oracle
    res = wss_oracle(args.case, args.t, sources=args.sources, frac=args.frac, seed=args.seed)
    path = _write_check(f"wss_case{args.case:02d}_t{args.t}", res)
    for r in res["rows"]:
        if r["estimator"] == "newton2" and "aneurysm" in r:
            print(f"[wss] {r['source']} h={r['h_mm']} mm: aneurysm |WSS| {r['aneurysm']['mag_rel_l2']:.3f}, "
                  f"whole wall {r['wall']['mag_rel_l2']:.3f}")
    print(f"[wss] wrote {path}")


# --------------------------------------------------------------------------- reconstructions
def cmd_baseline(args) -> None:
    from idealaorta_pinn.interpolation import run_comparator
    r = run_comparator(args.method, args.case, args.times, args.target, args.grid, args.seed,
                       min_wall_mm=args.min_wall_mm)
    print(f"[baseline] {r['run_name']}: velocity {r['velocity']['vel_rel_l2']:.4f}, "
          f"aneurysm WSS {r.get('wss', {}).get('wss_mag_rel_l2', float('nan')):.4f}, "
          f"pressure {r['pressure']['unsteady+mut']['p_rel_l2']:.4f} ({r['seconds']} s)")


def cmd_train(args) -> None:
    from idealaorta_pinn.training import ArmSpec, train_arm
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
    from idealaorta_pinn.training import diagnose
    for name in args.names:
        print(json.dumps(diagnose(name, device=args.device), indent=1, default=float))


# --------------------------------------------------------------------------- job queue
def cmd_jobs(args) -> None:
    from idealaorta_pinn.study import jobs, load_study
    print("\n".join(jobs(load_study(args.study), args.kind)))


def cmd_run_jobs(args) -> None:
    """Run each line of a job file as ``python main.py <line>``, ``--parallel`` at a time.

    GPU jobs are spread round-robin over ``--gpus``; ``--gpus`` with no value runs on CPU.
    One log per job goes to report/logs/, and one line per finished job, with its exit
    code, to report/logs/done.txt.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lines = [ln.strip() for ln in Path(args.jobfile).read_text().splitlines() if ln.strip()]
    done = LOG_DIR / "done.txt"
    running: list = []

    def reap() -> None:
        """Record and drop the finished jobs."""
        for item in list(running):
            p, line, log = item
            if p.poll() is not None:
                log.close()
                stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                with done.open("a") as f:
                    f.write(f"{stamp} exit={p.returncode} {line}\n")
                running.remove(item)

    for i, line in enumerate(lines):
        while len(running) >= args.parallel:
            time.sleep(5)
            reap()
        env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                   OPENBLAS_NUM_THREADS=str(args.threads),
                   CUDA_VISIBLE_DEVICES=args.gpus[i % len(args.gpus)] if args.gpus else "")
        tag = line.replace(" ", "_").replace("-", "")
        log = (LOG_DIR / f"{tag}.log").open("w")
        cmd = [sys.executable, "-u", "-W", "ignore", str(Path(__file__).resolve()), *line.split()]
        running.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), line, log))
        print(f"[run-jobs] {i + 1}/{len(lines)} {line}", flush=True)
    while running:
        time.sleep(5)
        reap()
    print(f"[run-jobs] finished; exit codes in {done}")


# --------------------------------------------------------------------------- analyses
def cmd_select(args) -> None:
    from idealaorta_pinn.study import load_study, select_weights
    out = select_weights(load_study(args.study))
    for row in out["table"]:
        print(f"{row['arm']:9s} {row['closure']:7s} w={row['wphys']:<5g} val_obs {row['val_obs_rel_l2']:.4f}")
    print("selected:", out["selected"])


def cmd_summarize(args) -> None:
    from idealaorta_pinn.study import load_study, summarize
    out = summarize(load_study(args.study))
    for unit, methods in out.items():
        print(f"\n{unit}")
        for m, v in methods.items():
            mean = lambda k: sum(v[k]) / len(v[k]) if v.get(k) else float("nan")  # noqa: E731
            print(f"  {m:24s} n={len(v['vel'])} vel {mean('vel'):.3f}  wss {mean('wss'):.3f}  "
                  f"p(unsteady+mut) {mean('p_unsteady+mut'):.3f}  p(own head) {mean('p_own'):.3f}")


def cmd_confirm(args) -> None:
    from idealaorta_pinn.study import confirm, load_study
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


def cmd_report(args) -> None:
    from idealaorta_pinn.study import load_study, report
    for kind, path in report(load_study(args.study)).items():
        print(f"[report] {kind}: {path}")


def cmd_figures(args) -> None:
    from idealaorta_pinn import plots
    for group in args.groups:
        for kind, path in {"cfd": plots.plot_cfd, "maps": plots.plot_maps}[group]().items():
            print(f"[figures] {kind}: {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sparse-velocity reconstruction study.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="raw CFD exports -> data/processed/full parquet")
    p.add_argument("--cases", type=int, nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("audit", help="momentum budget + pressure check of the CFD fields")
    p.add_argument("--case", type=int, required=True)
    p.add_argument("--t", type=int, required=True, help="snapshot (ms)")
    p.add_argument("--before", type=int, default=None, help="earlier snapshot for du/dt (ms)")
    p.add_argument("--after", type=int, default=None, help="later snapshot for du/dt (ms)")
    p.add_argument("--no-pressure", action="store_true")
    p.add_argument("--forms", nargs="*", default=None, help="subset of momentum forms for the pressure check")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("wss-oracle", help="wall-shear estimators from the CFD velocity")
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

    p = sub.add_parser("jobs", help="expand a study into command lines")
    p.add_argument("study", help="name under configs/reconstruction/ or a YAML path")
    p.add_argument("--kind", required=True, choices=["train", "baseline", "robustness", "reference"])
    p.set_defaults(func=cmd_jobs)

    p = sub.add_parser("run-jobs", help="run a file of command lines with bounded concurrency")
    p.add_argument("jobfile")
    p.add_argument("--parallel", type=int, default=4, help="jobs at a time")
    p.add_argument("--gpus", nargs="*", default=["0"], help="GPU ids for round-robin; none = CPU only")
    p.add_argument("--threads", type=int, default=2, help="BLAS/OpenMP threads per job")
    p.set_defaults(func=cmd_run_jobs)

    for name, func, help_ in (("select", cmd_select, "physics-weight selection table"),
                              ("summarize", cmd_summarize, "per-unit table of every method"),
                              ("confirm", cmd_confirm, "prespecified hypothesis tests"),
                              ("report", cmd_report, "CSV tables and result figures under report/")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("study", help="name under configs/reconstruction/ or a YAML path")
        p.set_defaults(func=func)

    p = sub.add_parser("figures", help="CFD-field figures and reconstruction maps under report/figures")
    p.add_argument("groups", nargs="+", choices=["cfd", "maps"])
    p.set_defaults(func=cmd_figures)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
