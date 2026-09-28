# IdealAorta-PINN

**A parametric physics-informed neural network (PINN) surrogate for idealized aortic aneurysm hemodynamics.**

This repository trains a single PINN that learns the hemodynamic field — velocity `(u, v, w)`,
pressure `p`, and wall shear stress (WSS) — as a function of **inlet diameter**, **saccular
aneurysm asymmetry**, and **cardiac phase**, for a family of idealized aortic geometries. The
network is trained against ANSYS CFD (incompressible RANS, SST k–ω transition turbulence model)
and validated by holding out entire inlet diameters (leave-one-diameter-out, LODO), so its
generalization to an unseen geometry can be measured directly against CFD.

It is the *idealized, parametric, generalizing* counterpart to the group's published
patient-specific work (Ur Rehman et al., *Physics of Fluids* 37(3):031913, 2025), and follows the
same modelling stack and PINN "house style" as the group's `TAA-aneurysm` and
`Double_pinn_hemodynamics` projects.

---

## Contents

- [Study design](#study-design)
- [Method](#method)
- [Data](#data)
- [Code layout](#code-layout)
- [Setup](#setup)
- [Pipeline](#pipeline)
- [Running at scale](#running-at-scale-multi-gpu--cluster)
- [Citation](#citation)

---

## Study design

12 idealized aortic geometries, built as a controlled `2 (health) x 3 (diameter) x symmetry` grid:

| Group      | Inlet Ø (cm)    | Asymmetry β = r/R                                     | Notes                                      |
|------------|-----------------|--------------------------------------------------------|---------------------------------------------|
| Aneurysmal | 2.0 / 2.3 / 2.6 | 1.0 (axisymmetric), 0.48 (anterior), 2.08 (posterior) | fixed 4.0 cm saccular bulge at mid-vessel  |
| Healthy    | 2.0 / 2.3 / 2.6 | — (smooth linear taper)                               | outlet = 80% of inlet diameter             |

Each case is exported by ANSYS CFD-Post at two phases of the cardiac cycle (systole, diastole) as:
3D velocity streamline point clouds, wall pressure + WSS point clouds, and XY/XZ mid-plane
velocity slices (used for validation and figures, never for training). `configs/cases.yaml` is the
authoritative per-case metadata (diameter, health, symmetry, β); `idealaorta_pinn/data/registry.py`
independently re-derives it from the (messy) on-disk folder names and cross-checks against it.

The core experiment design is **leave-one-diameter-out (LODO)**: train on two of the three inlet
diameters (all symmetry classes), hold out the third, and report error against CFD on the
held-out diameter. Three folds together cover every diameter — see
`configs/stageB_kfold_hold2p0.yaml`, `stageB_richerloo_f16.yaml` (holds 2.3 cm), and
`stageB_kfold_hold2p6.yaml`. `stageA_case*_insample.yaml` fit one surrogate per geometry
(the per-case reconstruction floor, one for each of the twelve CFD cases); the other
`stageA_*` configs are single-case de-risking runs.

## Method

- **Inputs:** Fourier-encoded spatial coordinates `(x, y, z)` ⊕ a small parameter encoder over
  `[d_inlet*, β, disease_flag, phase]`, concatenated before the manifold network
  (P²INN-style parametric conditioning — Cho et al. 2024; IP-PINN, Kalajahi-Arzani 2025).
- **Networks:** one decoupled, per-field residual-block Swish MLP for each of `u, v, w, p`, plus a
  smaller turbulent-viscosity network `ν_t(x; μ)` (softplus-positive output). Two AdamW
  optimizers train jointly, with `ν_t` at 10× the field networks' learning rate.
- **Physics:** steady RANS-mean momentum + continuity residual with effective viscosity
  `ν_eff = ν_mol + ν_t`, evaluated by autograd in standardized (non-dimensional) coordinates,
  solved quasi-steadily per cardiac phase (`idealaorta_pinn/pinn/physics.py`).
- **Data fit:** 3D CFD streamline velocity samples plus wall pressure/WSS point clouds
  (`idealaorta_pinn/pinn/losses.py`). WSS is computed from the autograd velocity gradient at the
  wall, not from a separate network. XY/XZ plane exports are reserved for validation and figures.
- **Loss balancing:** gradient-norm adaptive weighting (Wang et al. 2021) with an EMA, plus a
  per-phase velocity scale correction so systole and diastole receive an equal gradient budget
  despite their very different velocity magnitudes.
- **Validation:** leave-one-diameter-out vs. CFD — relative L2 and U_ref-normalized NRMSE on
  velocity, WSS error, wall-pressure pattern agreement (Pearson/Spearman on the mean-removed
  field, since the CFD pressure datum is arbitrary), and sac recirculation/swirl.

## Data

The CFD exports (raw CSVs **and** the parsed parquet caches) are **not tracked in this git
repository**; everything under `data/` is ignored except the two small index files below.
Two export vintages of the same rigid-wall CFD runs live under one tree:

```
data/
  raw/
    full_2026-09/Case <n>/<time>.csv     # whole-domain exports, five CFD-Post blocks per file:
                                        #   fluid volume, lumen wall (+ aneurysm zone), inlet, outlet;
                                        #   1.775-1.815 s (systole) and 2.39/2.4 s (diastole); ~8.4 GB
  processed/                            # parquet/npz cache of the original exports (3D streamlines,
                                        #   aneurysm-wall clip, XY/XZ planes, XY-plane x-WSS
                                        #   polylines); canonical, since those raw folders are
                                        #   archived off-repo
  processed/full/                       # per-block parquet of the whole-domain exports (README inside)
  registry.json                         # derived case index (tracked)
  results_on_slices.csv                 # slice-averaged CFD validation table (tracked)
```

Raw filenames are kept verbatim; their snapshot times are interpreted in
`idealaorta_pinn/data/full_export.py` (Case 1's `1.755.csv` is the 1.775 s snapshot).

## Code layout

```
idealaorta_pinn/
  config.py            # paths, physical constants, pulsatile inlet waveform
  data/
    registry.py         # discover data/raw/legacy_2025/Case* -> data/registry.json (case metadata)
    cfdpost.py           # ANSYS CFD-Post CSV parser (single- and multi-block exports)
    cache.py             # legacy CFD-Post CSV -> parquet/npz cache
    full_export.py       # whole-domain exports: raw CSV -> per-block parquet, snapshot loader
    normalize.py         # per-case/phase standardization (Normalizer)
    geometry.py           # wall-normal estimation (Open3D, PCA fallback) for WSS
  pinn/
    model.py             # Fourier-feature + parameter-encoder field networks
    physics.py             # RANS-mean momentum/continuity residuals (autograd)
    losses.py              # data-fidelity losses (velocity, pressure, autodiff WSS)
    train.py               # Trainer: optimization loop, loss balancing, checkpointing
  analysis/
    predict.py            # load a trained checkpoint, predict in physical units
    metrics.py              # error metrics vs CFD; cross-run evaluation + LODO table
    figures.py               # static (matplotlib) + interactive (Plotly 3D) figures
    rescore.py               # re-score runs on every point set beside interpolation baselines
    full_scoring.py          # score models and baselines on the whole-vessel exports
    consistency.py           # divergence / no-slip / pressure consistency vs interpolation
    hidden_fields.py         # wall shear and pressure inferred from velocity-only models
  reconstruction/        # sparse-observation reconstruction study (see below)
    numerics.py            # least-squares node gradients, sparse gradient operator
    problem.py             # one problem: case, time window, observation grid, hidden targets
    fields.py              # space-time neural field + RANS-mean residuals
    postprocess.py         # velocity -> pressure (momentum integration) and wall shear, scores
    interpolation.py       # linear / RBF / tuned / space-time RBF comparators
    reference.py           # training-free audits: momentum budget, pressure and WSS oracles
    training.py            # fit and score one neural-field arm
    study.py               # study configs -> jobs; selection, summaries, hypothesis tests

scripts/
  prepare.py              # registry / cache / full subcommands (data pipeline)
  run.py                  # train + validate + figures for one config (the main entry point)
  report.py               # post-training: evaluate / kfold-table / error-vs-diameter /
                          #   rescore / score-full / consistency / infer-hidden
  reconstruct.py          # reconstruction study: audit / wss-oracle / baseline / train /
                          #   diagnose / jobs / select / summarize / confirm
  manuscript/             # figure/table generators for the September 2026 manuscript
  queue.sh                # run configs back-to-back, each gated on real GPU headroom
  queue_jobs.sh           # run a file of reconstruct.py jobs with bounded concurrency
  sulis/                  # setup.sh (conda env on Sulis) and array.slurm (one job line per task)
  regen_interactive.sh    # rebuild the rotatable 3D HTML for every run (inference only)

configs/                  # one YAML per experiment (stageA_case*_insample per-case fits,
                           # other stageA_* de-risking, stageB_* LODO folds, diag_* / lodo_full_*
                           # whole-export runs), cases/constants.yaml, and reconstruction/*.yaml
tests/                    # pytest: residual derivatives, study plumbing, data conversion
```

## Setup

Python 3.11 + PyTorch (CUDA build matching your GPU) plus the packages in `requirements.txt`.
Each script inserts the repo root onto `sys.path`, so nothing needs to be installed as a package:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your CUDA version
pip install -r requirements.txt
```

`open3d` is an optional dependency (`pip install open3d`) for higher-quality mesh wall-normal
estimation; the code falls back to a PCA-based normal estimator (via scikit-learn) if it isn't
installed or fails to import.

## Pipeline

```bash
# 1) data prep (once data/raw/ is populated -- see "Data" above)
python scripts/prepare.py registry             # discover legacy cases -> data/registry.json
python scripts/prepare.py cache                # parse legacy CFD-Post CSVs -> data/processed/*.parquet
python scripts/prepare.py full                 # whole-domain CSVs -> data/processed/full/*.parquet
#    (or: scripts/prepare.py all  -- registry, cache, then full)

# 2) train + validate + generate figures (one workflow, since they share the trained model)
python scripts/run.py --config configs/stageA_case1.yaml                       # Stage A de-risk (Case 1)
python scripts/run.py --config configs/stageB_richerloo_f16.yaml --cases 4 5 6 # Stage B: predict unseen 2.3 cm (LODO)
python scripts/run.py --config configs/stageA_case1.yaml --skip-train          # re-validate/plot an existing model

# 3) post-processing across multiple trained runs (no retraining; --device cpu works)
python scripts/report.py evaluate            # score every trained run vs CFD -> report/metrics/all_runs.json
python scripts/report.py kfold-table         # LODO generalization table -> report/tables/ (md + csv + tex)
python scripts/report.py error-vs-diameter   # held-out error vs in-sample floor -> report/figures/
```

To run several configs back-to-back on one GPU, `scripts/queue.sh` waits for real free memory
before each job rather than racing an in-flight run into an OOM:

```bash
scripts/queue.sh stageA_case5_insample stageA_case6_insample      # ~12 GB each
NEED_MIB=26000 scripts/queue.sh stageB_kfold_hold2p0              # a LODO fold, ~23 GB
```

`scripts/regen_interactive.sh` rebuilds the rotatable 3D HTML for every run straight from the
saved checkpoints — use it if a batch was launched with `--no-interactive`.

Outputs land in experiment-scoped folders: `models/<experiment>/` for checkpoints and training
history, `report/{figures,metrics,tables,interactive,logs}/<experiment>/` for generated outputs.
`report/figures/` is regenerable and gitignored; if you want to curate a specific figure for
publication, copy it out deliberately rather than committing the whole directory. Use
`--device cuda` (default) and, if you hit memory limits, lower `loaders.max_velocity_points` or
`physics.n_collocation` in the config.

### Sparse-reconstruction study

Recovers hidden velocity, gauge pressure and aneurysm-wall shear from velocity sampled on a
grid at two nearby instants, and compares space-time neural fields (with and without physics
in the loss) against interpolation, all scored through the same velocity-to-pressure and
wall-shear pipeline. Studies are defined in `configs/reconstruction/`; results go to
`report/metrics/reconstruction/` and checkpoints to `models/rev2_*`.

```bash
python scripts/reconstruct.py audit --case 1 --t 1780          # do the CFD fields satisfy the momentum forms?
python scripts/reconstruct.py jobs confirm --kind train > jobs_train.txt
python scripts/reconstruct.py jobs confirm --kind baseline > jobs_baseline.txt
scripts/queue_jobs.sh jobs_train.txt 6 gpu                    # GPUs from $GPUS (default "0 1")
scripts/queue_jobs.sh jobs_baseline.txt 3 cpu
python scripts/reconstruct.py confirm confirm                  # pre-registered hypothesis tests
python -m pytest tests                                         # derivative and plumbing checks
```

When re-running an experiment that already has outputs, `scripts/run.py` archives the existing
model/report folders into sibling `_archive/` directories before starting the new run. Pass
`--overwrite-output` to replace outputs in place instead, or `--seed N` / `--name-suffix <tag>` to
run a variant (e.g. a seed sweep) into its own `models/<name>_<tag>/` without touching the base run.

## Running at scale (multi-GPU / cluster)

The LODO folds train one network on six geometries at once and are the most memory-hungry
runs (~23 GB); the per-case `stageA_case*_insample` fits are light (~12 GB each). All fit on
a single large-memory GPU. The project is portable — no scheduler script needed; SSH in and
run interactively under `tmux` so long runs survive disconnects:

```bash
git clone <this-repo-url>
cd IdealAorta-PINN

# environment (see "Setup" above), then match torch to the node's CUDA version, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# copy data/raw/ (or the original CFD exports) onto the node, then:
python scripts/prepare.py all

tmux new -s ideal
python scripts/run.py --config configs/stageB_richerloo_f16.yaml   # a LODO fold (train 6, predict held-out diameter)
#   detach: Ctrl-b then d;  reattach: tmux attach -t ideal
```

### Sulis (Slurm)

The reconstruction study is many small independent jobs (~2.5 GB GPU memory, 10-110 min
each), which suits a Slurm job array on Sulis's L40 nodes (account `su003-csmm`). The code
arrives by `git clone`; the gitignored inputs are copied once from a machine that has them:

```bash
# from the local machine (one 2FA prompt): whole-domain cache, registry, study references
tar -cf - data/processed/full data/registry.json report/metrics/reconstruction   | ssh sulis "mkdir -p ~/IdealAorta-PINN && tar -xf - -C ~/IdealAorta-PINN"

# on Sulis
bash scripts/sulis/setup.sh                                    # once: conda env "idealaorta"
python scripts/reconstruct.py jobs confirm --kind train > jobs_train.txt
mkdir -p report/logs/reconstruction/slurm
sbatch --array=1-$(grep -c . jobs_train.txt)%40 scripts/sulis/array.slurm jobs_train.txt
squeue -u $USER                                                # progress; results as on brosnan
```

## Citation

If you use this code or the accompanying dataset, please cite the companion patient-specific
study this project extends:

> Ur Rehman et al., "[title]," *Physics of Fluids* 37(3):031913, 2025.
