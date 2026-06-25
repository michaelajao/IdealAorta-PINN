"""Probe centerline velocity-peak fidelity across trained models.

The velocity NRMSE and WSS metrics do not expose the centerline-jet *peak*
over-prediction that motivated the Fourier-bandwidth sweep. This prints, per
model and phase, max(predicted speed) / max(CFD speed) on the validation slice —
the peak ratio (1.0 = perfect; >1 = over-prediction). Read-only; loads each
best_model.pt and predicts on the CFD slice.

Usage:
    python scripts/peak_probe.py --experiments stageA_case1_s12 --case 1 --kind XZ --device cuda
    python scripts/peak_probe.py --experiments stageB_richerloo_f16 --case 4 --kind XZ --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idealaorta_pinn.config import MODELS_DIR  # noqa: E402
from idealaorta_pinn.data.cache import load_points  # noqa: E402
from idealaorta_pinn.data.registry import cases_by_id, load_registry  # noqa: E402
from idealaorta_pinn.analysis.predict import load_trained, predict_physical  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", required=True)
    ap.add_argument("--case", type=int, default=1)
    ap.add_argument("--kind", type=str, default="XZ")
    ap.add_argument("--phases", nargs="+", default=["systolic", "diastolic"])
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    records = load_registry()
    rec = cases_by_id(records)[args.case]
    beta = rec.beta if rec.beta is not None else 1.0

    print(f"{'experiment':<26} {'phase':<10} {'CFD_peak':>9} {'PINN_peak':>10} {'ratio':>7}")
    print("-" * 66)
    for exp in args.experiments:
        ckpt = MODELS_DIR / exp / "best_model.pt"
        if not ckpt.exists():
            print(f"{exp:<26} (no best_model.pt)")
            continue
        model = load_trained(ckpt, device=args.device)
        for phase in args.phases:
            df = load_points(rec, args.kind, phase)
            if df is None or not {"u", "v", "w"}.issubset(df.columns):
                continue
            coords = df[["x", "y", "z"]].to_numpy(float)
            true = df[["u", "v", "w"]].to_numpy(float)
            pred = predict_physical(model, coords, rec.inlet_diameter_cm,
                                    rec.disease_flag, phase, beta=beta)
            cfd_peak = float(np.linalg.norm(true, axis=1).max())
            pinn_peak = float(np.asarray(pred["speed"]).max())
            ratio = pinn_peak / cfd_peak if cfd_peak else float("nan")
            print(f"{exp:<26} {phase:<10} {cfd_peak:>9.3f} {pinn_peak:>10.3f} {ratio:>7.2f}")


if __name__ == "__main__":
    main()
