"""Run a stage end to end: train, validate against CFD, and generate figures.

Training, validation and figure generation are one workflow because they share
the trained model. Use ``--skip-train`` to re-validate / re-plot an existing
checkpoint, and ``--cases`` to target held-out cases (e.g. the 2.3 cm
leave-one-diameter-out case in Stage B).

Usage:
    python scripts/run.py --config configs/stageA_case1.yaml
    python scripts/run.py --config configs/stageB_loo_2p3.yaml --cases 4
    python scripts/run.py --config configs/stageA_case1.yaml --skip-train --no-interactive
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idealaorta_pinn.config import (FIGURES_DIR, INTERACTIVE_DIR, LOGS_DIR, METRICS_DIR,  # noqa: E402
                                    PAPER_FIGURES_DIR, PROJECT_ROOT, TABLES_DIR, load_yaml)
from idealaorta_pinn.data.registry import cases_by_id, load_registry  # noqa: E402


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        self.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _has_artifacts(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return True
    return any(path.iterdir())


def _archive_existing(path: Path, stamp: str) -> Path | None:
    if not _has_artifacts(path):
        return None
    archive_root = path.parent / "_archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    dest = archive_root / f"{path.name}_{stamp}"
    i = 1
    while dest.exists():
        dest = archive_root / f"{path.name}_{stamp}_{i}"
        i += 1
    path.rename(dest)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="Train + validate + figures for one stage.")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--skip-train", action="store_true", help="Reuse the existing checkpoint.")
    ap.add_argument("--cases", type=int, nargs="*", default=None,
                    help="Cases to validate/plot (default: the config's train_cases).")
    ap.add_argument("--phases", type=str, nargs="*", default=None)
    ap.add_argument("--val-kind", type=str, default="XZ", help="CFD slice for velocity error.")
    ap.add_argument("--fig-kind", type=str, default="XY", help="Plane for static figures.")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-interactive", action="store_true")
    ap.add_argument("--no-copy-paper", action="store_true", help="Do not copy PNGs to paper/figures.")
    ap.add_argument("--no-run-log", action="store_true",
                    help="Do not mirror stdout/stderr to report/logs/<experiment>/run.log.")
    ap.add_argument("--overwrite-output", action="store_true",
                    help="Overwrite existing experiment outputs instead of archiving them before training.")
    ap.add_argument("--paper-reference-table", action="store_true",
                    help="Write cleaned results_on_slices.csv reference table.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    cfg["device"] = args.device
    name = cfg["experiment"]["name"]
    out_dir = _resolve(cfg.get("output_dir", f"models/{name}"))
    fig_dir = FIGURES_DIR / name
    paper_dir = PAPER_FIGURES_DIR / name
    inter_dir = INTERACTIVE_DIR / name
    metrics_dir = METRICS_DIR / name
    tables_dir = TABLES_DIR / name
    log_dir = LOGS_DIR / name

    archived = []
    if not args.skip_train and not args.overwrite_output:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for path in (out_dir, fig_dir, paper_dir, inter_dir, metrics_dir, tables_dir, log_dir):
            dest = _archive_existing(path, stamp)
            if dest is not None:
                archived.append((path, dest))

    log_file = None
    if not args.no_run_log:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(log_dir / "run.log", "w", encoding="utf-8", buffering=1)
        sys.stdout = _Tee(sys.__stdout__, log_file)
        sys.stderr = _Tee(sys.__stderr__, log_file)
        print(f"[run] logging to {(log_dir / 'run.log').relative_to(PROJECT_ROOT)}")
    for old, new in archived:
        print(f"[run] archived {old.relative_to(PROJECT_ROOT)} -> {new.relative_to(PROJECT_ROOT)}")

    # ---- train ----
    if not args.skip_train:
        from idealaorta_pinn.pinn.train import Trainer
        Trainer(cfg).train()

    # ---- load trained model ----
    from idealaorta_pinn.analysis.predict import load_trained
    from idealaorta_pinn.analysis.metrics import velocity_metrics, wss_metrics, write_report
    model = load_trained(out_dir / "best_model.pt", device=args.device)
    records = load_registry()
    by_id = cases_by_id(records)
    cases = args.cases or list(cfg["data"]["train_cases"])
    phases = args.phases or list(cfg["data"].get("phases", ["systolic", "diastolic"]))

    # ---- validate ----
    vel_rows, wss_rows = [], []
    for cid in cases:
        for ph in phases:
            vm = velocity_metrics(model, records, cid, ph, kind=args.val_kind)
            if vm:
                vm["diameter_cm"] = by_id[cid].inlet_diameter_cm
                vel_rows.append(vm)
            wm = wss_metrics(model, records, cid, ph)
            if wm:
                wm["diameter_cm"] = by_id[cid].inlet_diameter_cm
                wss_rows.append(wm)
    write_report(vel_rows, "velocity", f"{name}: velocity error ({args.val_kind})",
                 out_dir=metrics_dir)
    write_report(wss_rows, "wss", f"{name}: WSS error", out_dir=metrics_dir)

    print(f"\n=== {name}: velocity vs CFD ({args.val_kind}) ===")
    for r in vel_rows:
        print(f"  case {r['case']} {r['phase']:<9} rel-L2={r['vel_rel_l2']:.4f}  "
              f"nrmse/Uref={r['vel_nrmse_uref']:.4f}  "
              f"recirc CFD/PINN={r['recirc_cfd']:.3f}/{r['recirc_pinn']:.3f}")
    for r in wss_rows:
        print(f"  case {r['case']} {r['phase']:<9} WSS rel-L2={r['wss_rel_l2']:.4f}  "
              f"peak {r['wss_peak_cfd']:.1f}/{r['wss_peak_pinn']:.1f} Pa")

    # ---- figures ----
    # Outputs are namespaced per experiment so different stages on the same case
    # don't overwrite each other's plots (figures are keyed by case/phase only).
    if not args.no_figures:
        from idealaorta_pinn.analysis.figures import plane_comparison, convergence_curves
        from idealaorta_pinn.analysis.figures_paper import (wss_map, wall_pressure_map,
                                                            velocity_profile, axial_velocity_profile)
        produced = []
        for cid in cases:
            for ph in phases:
                # velocity field comparison (CFD | PINN | error) in BOTH planes
                for kind in ("XY", "XZ"):
                    try:
                        produced.append(plane_comparison(model, records, cid, ph, kind=kind,
                                                         shared_scale=True, out_dir=fig_dir))
                    except Exception as e:  # noqa: BLE001
                        print(f"  [figure] skip {kind} velocity case {cid} {ph}: {e}")
                # WSS map, wall pressure, transverse bulge profile, axial profile
                for label, fn in (("wss-map", wss_map), ("wall-pressure", wall_pressure_map),
                                  ("bulge-profile", velocity_profile),
                                  ("axial-profile", axial_velocity_profile)):
                    try:
                        produced.append(fn(model, records, cid, ph, out_dir=fig_dir))
                    except Exception as e:  # noqa: BLE001
                        print(f"  [figure] skip {label} case {cid} {ph}: {e}")
        # training convergence curve (once per run, from loss_history.csv)
        hist = out_dir / "loss_history.csv"
        if hist.exists():
            try:
                produced.append(convergence_curves(hist, out_dir=fig_dir, title=f"{name} — convergence"))
            except Exception as e:  # noqa: BLE001
                print(f"  [figure] skip convergence: {e}")
        if not args.no_copy_paper:
            paper_dir.mkdir(parents=True, exist_ok=True)
            for p in produced:
                shutil.copy2(p, paper_dir / p.name)
        print(f"[figures] wrote {len(produced)} PNG(s) to report/figures/{name}"
              f"{f' (+ paper/figures/{name})' if not args.no_copy_paper else ''}")

        if args.paper_reference_table:
            from idealaorta_pinn.analysis.figures_paper import slice_reference_table
            try:
                t = slice_reference_table(out_dir=tables_dir)
                print(f"[figures] wrote {t.relative_to(PROJECT_ROOT)}")
            except Exception as e:  # noqa: BLE001
                print(f"  [paper-reference-table] skipped: {e}")

    # ---- interactive ----
    if not args.no_interactive:
        from idealaorta_pinn.analysis.figures import save_comparison
        for cid in cases:
            for ph in phases:
                try:
                    save_comparison(model, records, cid, ph, out_dir=inter_dir)
                except Exception as e:  # noqa: BLE001
                    print(f"  [interactive] skip case {cid} {ph}: {e}")
        print(f"[interactive] wrote rotatable HTML to report/interactive/{name}")


if __name__ == "__main__":
    main()
