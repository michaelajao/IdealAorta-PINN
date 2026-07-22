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
| Aneurysmal | 2.0 / 2.3 / 2.6 | 1.0 (axisymmetric), 2.08 (anterior), 0.48 (posterior) | fixed 4.0 cm saccular bulge at mid-vessel  |
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
`stageB_kfold_hold2p6.yaml`. `stageA_*` configs are single-case de-risking runs; `stageC_*` trains
on the full 12-case study.

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

The 12 CFD cases (raw CFD-Post exports **and** the parsed parquet/npz cache) are **not tracked in
this git repository** — at ~1.1 GB raw + ~0.2 GB processed they don't belong in git history.

<!-- TODO: link the permanent data repository (e.g. Zenodo/OSF DOI) once the dataset is deposited. -->

Once obtained, place the 12 `Case *` folders (as exported by ANSYS CFD-Post) under `data/raw/`,
matching the layout `data/raw/Case <n>, <diameter> Inlet ... /PINNS/*.csv`, then run the data
pipeline below to build the registry and parquet cache. `data/registry.json` (the small, derived
case index) and `data/results_on_slices.csv` (an independent slice-averaged CFD validation table)
are the only data artifacts tracked in git.

## Code layout

```
idealaorta_pinn/
  config.py            # paths, physical constants, pulsatile inlet waveform
  data/
    registry.py         # discover data/raw/Case* -> data/registry.json (case metadata)
    cfdpost.py           # robust ANSYS CFD-Post CSV parser (column + duplicate handling)
    cache.py             # CFD-Post CSV -> parquet/npz cache
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

scripts/
  prepare.py              # registry / cache subcommands (data pipeline)
  run.py                  # train + validate + figures for one config (the main entry point)
  report.py               # post-training: evaluate / kfold-table / error-vs-diameter

configs/                  # one YAML per experiment (stageA_* de-risking, stageB_* LODO
                           # folds, stageC_* full study), plus cases.yaml / constants.yaml
```

## Setup

Python 3.11 + PyTorch (CUDA build matching your GPU) plus a short list of standard scientific
packages — there is no `pyproject.toml`/`requirements.txt`; each script inserts the repo root onto
`sys.path`, so nothing needs to be installed as a package:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA version
pip install numpy pandas pyyaml scipy scikit-learn matplotlib plotly
```

`open3d` is an optional dependency (`pip install open3d`) for higher-quality mesh wall-normal
estimation; the code falls back to a PCA-based normal estimator (via scikit-learn) if it isn't
installed or fails to import.

## Pipeline

```bash
# 1) data prep (once data/raw/ is populated -- see "Data" above)
python scripts/prepare.py registry             # discover cases -> data/registry.json
python scripts/prepare.py cache                # parse CFD-Post CSVs -> data/processed/*.parquet
#    (or: scripts/prepare.py all  -- registry, then cache, in one go)

# 2) train + validate + generate figures (one workflow, since they share the trained model)
python scripts/run.py --config configs/stageA_case1.yaml                       # Stage A de-risk (Case 1)
python scripts/run.py --config configs/stageB_richerloo_f16.yaml --cases 4 5 6 # Stage B: predict unseen 2.3 cm (LODO)
python scripts/run.py --config configs/stageA_case1.yaml --skip-train          # re-validate/plot an existing model

# 3) post-processing across multiple trained runs (no retraining; --device cpu works)
python scripts/report.py evaluate            # score every trained run vs CFD -> report/metrics/all_runs.json
python scripts/report.py kfold-table         # LODO generalization table -> report/tables/ (md + csv + tex)
python scripts/report.py error-vs-diameter   # held-out error vs in-sample floor -> report/figures/
```

Outputs land in experiment-scoped folders: `models/<experiment>/` for checkpoints and training
history, `report/{figures,metrics,tables,interactive,logs}/<experiment>/` for generated outputs.
`report/figures/` is regenerable and gitignored; if you want to curate a specific figure for
publication, copy it out deliberately rather than committing the whole directory. Use
`--device cuda` (default) and, if you hit memory limits, lower `loaders.max_velocity_points` or
`physics.n_collocation` in the config.

When re-running an experiment that already has outputs, `scripts/run.py` archives the existing
model/report folders into sibling `_archive/` directories before starting the new run. Pass
`--overwrite-output` to replace outputs in place instead, or `--seed N` / `--name-suffix <tag>` to
run a variant (e.g. a seed sweep) into its own `models/<name>_<tag>/` without touching the base run.

## Running at scale (multi-GPU / cluster)

The heavier configs (`stageC_*`: the full 12-case study, larger networks, more collocation points)
are best run on a large-memory GPU. The project is portable — no scheduler script needed; SSH in
and run interactively under `tmux` so long runs survive disconnects:

```bash
git clone <this-repo-url>
cd IdealAorta-PINN

# environment (see "Setup" above), then match torch to the node's CUDA version, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# copy data/raw/ (or the original CFD export) onto the node, then:
python scripts/prepare.py registry
python scripts/prepare.py cache

tmux new -s ideal
python scripts/run.py --config configs/stageC_hpc.yaml   # full study, scaled for a large GPU
#   detach: Ctrl-b then d;  reattach: tmux attach -t ideal
```

## Citation

If you use this code or the accompanying dataset, please cite the companion patient-specific
study this project extends:

> Ur Rehman et al., "[title]," *Physics of Fluids* 37(3):031913, 2025.

<!-- TODO: add a citation entry (and CITATION.cff) for this project's own paper once published. -->
