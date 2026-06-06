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
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idealaorta_pinn.config import PAPER_FIGURES_DIR, PROJECT_ROOT, load_yaml  # noqa: E402
from idealaorta_pinn.data.registry import cases_by_id, load_registry  # noqa: E402


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


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
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    cfg["device"] = args.device
    name = cfg["experiment"]["name"]
    out_dir = _resolve(cfg.get("output_dir", f"models/{name}"))

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
    write_report(vel_rows, f"{name}_velocity", f"{name}: velocity error ({args.val_kind})")
    write_report(wss_rows, f"{name}_wss", f"{name}: WSS error")

    print(f"\n=== {name}: velocity vs CFD ({args.val_kind}) ===")
    for r in vel_rows:
        print(f"  case {r['case']} {r['phase']:<9} rel-L2={r['vel_rel_l2']:.4f}  "
              f"nrmse/Uref={r['vel_nrmse_uref']:.4f}  "
              f"recirc CFD/PINN={r['recirc_cfd']:.3f}/{r['recirc_pinn']:.3f}")
    for r in wss_rows:
        print(f"  case {r['case']} {r['phase']:<9} WSS rel-L2={r['wss_rel_l2']:.4f}  "
              f"peak {r['wss_peak_cfd']:.1f}/{r['wss_peak_pinn']:.1f} Pa")

    # ---- figures ----
    if not args.no_figures:
        from idealaorta_pinn.analysis.figures import error_vs_diameter, plane_comparison
        produced = []
        for cid in cases:
            for ph in phases:
                try:
                    produced.append(plane_comparison(model, records, cid, ph, kind=args.fig_kind))
                except Exception as e:  # noqa: BLE001
                    print(f"  [figure] skip case {cid} {ph}: {e}")
        if vel_rows:
            produced.append(error_vs_diameter(vel_rows))
        if not args.no_copy_paper:
            PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            for p in produced:
                shutil.copy2(p, PAPER_FIGURES_DIR / p.name)
        print(f"[figures] wrote {len(produced)} PNG(s) to report/figures"
              f"{' (+ paper/figures)' if not args.no_copy_paper else ''}")

    # ---- interactive ----
    if not args.no_interactive:
        from idealaorta_pinn.analysis.figures import save_comparison
        for cid in cases:
            for ph in phases:
                try:
                    save_comparison(model, records, cid, ph)
                except Exception as e:  # noqa: BLE001
                    print(f"  [interactive] skip case {cid} {ph}: {e}")
        print("[interactive] wrote rotatable HTML to report/interactive")


if __name__ == "__main__":
    main()
