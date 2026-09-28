"""Post-training reporting CLI: evaluate trained runs, build the LODO table and figure.

All three subcommands are pure post-processing over runs that already exist -- train
first with ``scripts/run.py``. The logic lives in the analysis package
(``analysis.metrics`` for the evaluation and the table, ``analysis.figures`` for the
figure); this file is only the command-line front end.

Subcommands
-----------
evaluate            Score every reported run against CFD from its saved checkpoint
                    (velocity, WSS, pressure, continuity) into report/metrics/all_runs.json.
                    Inference only; a run that has not been trained is skipped with a note.
kfold-table         Leave-one-diameter-out generalization table -> report/tables/ as
                    Markdown + CSV + LaTeX.
error-vs-diameter   Held-out error against the in-sample floor -> report/figures/.
rescore             Re-score every reported run on XY, XZ and 3D points with the full
                    error vector, beside the CFD-interpolation baselines
                    (``analysis.rescore``) -> report/metrics/_rescore, report/tables/.
score-full          Score models and baselines on the whole-vessel exports: test cubes,
                    slabs, whole-wall WSS, eddy viscosity (``analysis.full_scoring``).
consistency         Physical consistency (divergence, no-slip, pressure) of models vs
                    interpolation on held-out cases (``analysis.consistency``).
infer-hidden        Wall shear and pressure a velocity-only model infers, vs interpolated
                    velocity (``analysis.hidden_fields``).

Usage:
    python scripts/report.py evaluate
    python scripts/report.py evaluate --device cpu --runs lodo_2.0 lodo_2.3
    python scripts/report.py kfold-table --folds \
        2.0:stageB_kfold_hold2p0 2.3:stageB_richerloo_f16 2.6:stageB_kfold_hold2p6
    python scripts/report.py error-vs-diameter --folds \
        2.0:stageB_kfold_hold2p0 2.3:stageB_richerloo_f16 2.6:stageB_kfold_hold2p6 \
        --insample stageA_case1_s12

Note the folds must not be averaged into a single number: 2.3 cm is interpolation
between the training diameters, 2.0 and 2.6 cm are extrapolation past a single
bracketing diameter, and each fold carries its own train-set nondimensionalization.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idealaorta_pinn.analysis import consistency, figures, full_scoring, hidden_fields, metrics, rescore  # noqa: E402

_DEFAULT_FOLDS = ["2.0:stageB_kfold_hold2p0",
                  "2.3:stageB_richerloo_f16",
                  "2.6:stageB_kfold_hold2p6"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Post-training reporting (metrics, tables, summary figure).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_h = sub.add_parser("evaluate", help="score trained runs against CFD")
    p_h.add_argument("--device", default="cuda", help="torch device (cuda/cpu)")
    p_h.add_argument("--runs", nargs="*", default=None,
                     help="subset of run labels (default: all)")
    p_h.add_argument("--out", type=Path, default=None,
                     help="output JSON (default: report/metrics/all_runs.json)")

    p_k = sub.add_parser("kfold-table", help="LODO generalization table")
    p_k.add_argument("--folds", nargs="+", default=_DEFAULT_FOLDS,
                     help='"held_cm:experiment" specs, in display order')
    p_k.add_argument("--phases", nargs="*", default=list(metrics.PHASES))
    p_k.add_argument("--out", default="lodo_kfold", help="basename under report/tables/")
    p_k.add_argument("--title",
                     default="Leave-one-diameter-out generalization (per-phase NRMSE)")

    p_e = sub.add_parser("error-vs-diameter", help="held-out error vs in-sample floor")
    p_e.add_argument("--folds", nargs="+", default=_DEFAULT_FOLDS,
                     help='"held_cm:experiment" specs')
    p_e.add_argument("--insample", default="stageA_case1_s12",
                     help="experiment providing the in-sample floor line")
    p_e.add_argument("--out", default="error_vs_diameter",
                     help="basename under report/figures/")

    # full tools with their own options: each module owns add_arguments / run
    for name, module, help_ in (("rescore", rescore, "re-score runs on every point set"),
                                ("score-full", full_scoring, "score on the whole-vessel exports"),
                                ("consistency", consistency, "physical consistency vs interpolation"),
                                ("infer-hidden", hidden_fields, "hidden WSS / pressure from velocity")):
        p = sub.add_parser(name, help=help_, description=module.__doc__.splitlines()[0])
        module.add_arguments(p)
        p.set_defaults(run=module.run)

    args = ap.parse_args()
    if hasattr(args, "run"):
        args.run(args)
    elif args.cmd == "evaluate":
        metrics.evaluate_runs(runs=args.runs, device=args.device, out=args.out)
    elif args.cmd == "kfold-table":
        metrics.kfold_table(folds=args.folds, phases=args.phases, out=args.out,
                            title=args.title)
    elif args.cmd == "error-vs-diameter":
        figures.error_vs_diameter(folds=args.folds, insample=args.insample, out=args.out)


if __name__ == "__main__":
    main()
