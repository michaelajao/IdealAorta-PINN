# IdealAorta-PINN

**Recovering hidden pressure and wall shear stress from sparse velocity in idealized descending
thoracic aortic aneurysms.**

This repository holds the reconstruction study of the paper *Hemodynamics of Idealized Descending
Thoracic Aortic Aneurysms and Physics-Based Recovery of Pressure and Wall Shear Stress from Sparse
Velocity*. Given velocity observed at about 1.3% of the nodes of a rigid-wall CFD solution at two
nearby instants, it recovers the velocity elsewhere, the gauge pressure and the wall shear stress
(WSS) on the aneurysm zone. Space-time neural fields (with and without a continuity penalty, and
PINNs with the momentum balance in the loss) are compared with linear and RBF interpolation, and
every method passes through the same momentum-integration and near-wall shear calculation, so the
comparison isolates the reconstruction itself.

The original parametric-PINN study (leave-one-diameter-out surrogate, submitted to *Physics of
Fluids*) is preserved at the git tag [`pof-pinn-study`](../../tree/pof-pinn-study).

---

## Contents

- [Geometries and data](#geometries-and-data)
- [Code layout](#code-layout)
- [Setup](#setup)
- [Pipeline](#pipeline)
- [Running on a Slurm cluster](#running-on-a-slurm-cluster)
- [Citation](#citation)

---

## Geometries and data

Twelve idealized descending thoracic geometries, simulated as transient rigid-wall CFD with the
transition SST k–ω model:

| Group      | Inlet Ø (cm)    | Asymmetry β = r/R                                     | Notes                                    |
|------------|-----------------|--------------------------------------------------------|------------------------------------------|
| Aneurysmal | 2.0 / 2.3 / 2.6 | 1.0 (axisymmetric), 0.48 (anterior), 2.08 (posterior) | one dilation of fixed 4.0 cm maximum diameter |
| Control    | 2.0 / 2.3 / 2.6 | — (smooth linear taper)                               | outlet = 80% of inlet diameter           |

The CFD exports are **not tracked in this repository**; everything under `data/` is ignored except
the small slice summary:

```
data/
  raw/full_2026-09/Case <n>/<time>.csv   # whole-domain CFD-Post exports, five blocks per file:
                                        #   fluid volume, lumen wall (+ aneurysm zone), inlet, outlet;
                                        #   1.775-1.815 s (systole) and 2.39/2.4 s (diastole)
  processed/full/                       # per-block parquet of those exports (README inside)
  results_on_slices.csv                 # plane-averaged speed and k on planes D1-D8 (tracked)
```

Raw filenames are kept verbatim; their snapshot times are interpreted in
`idealaorta_pinn/data/full_export.py` (Case 1's `1.755.csv` is the 1.775 s snapshot).

## Code layout

```
idealaorta_pinn/
  config.py              # paths and constants loader
  data/
    cfdpost.py           # ANSYS CFD-Post CSV parser (single- and multi-block exports)
    full_export.py       # whole-domain exports: raw CSV -> per-block parquet, snapshot loader
  reconstruction/
    numerics.py          # least-squares node gradients, sparse gradient operator
    problem.py           # one problem: case, time window, observation grid, hidden targets
    fields.py            # space-time neural field + RANS-mean residuals
    postprocess.py       # velocity -> pressure (momentum integration) and wall shear, scores
    interpolation.py     # linear / RBF / tuned / space-time RBF comparators
    reference.py         # training-free audits: momentum budget, pressure and WSS oracles
    training.py          # fit and score one neural-field arm
    study.py             # study configs -> jobs; selection, summaries, hypothesis tests

scripts/
  prepare.py             # raw whole-domain CSVs -> data/processed/full
  reconstruct.py         # audit / wss-oracle / baseline / train / diagnose / jobs /
                         #   select / summarize / confirm
  queue_jobs.sh          # run a file of reconstruct.py jobs with bounded concurrency
  manuscript/            # figure and table generators for the manuscript (README inside)
  sulis/                 # setup.sh (conda env) and array.slurm (one job line per task)

configs/
  constants.yaml         # fluid properties, waveform and pressure constants
  reconstruction/        # pilot, selection, confirmation and reference study definitions
tests/                   # pytest: residual derivatives and study plumbing
```

## Setup

Python 3.11 + PyTorch (CUDA build matching your GPU) plus the packages in `requirements.txt`.
Each script inserts the repo root onto `sys.path`, so nothing needs to be installed as a package:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your CUDA version
pip install -r requirements.txt
```

## Pipeline

```bash
# 1) data prep, once data/raw/full_2026-09/ is populated
python scripts/prepare.py full                 # -> data/processed/full/*.parquet

# 2) reconstruction study
python scripts/reconstruct.py audit --case 1 --t 1780          # do the CFD fields satisfy the momentum forms?
python scripts/reconstruct.py jobs confirm --kind train > jobs_train.txt
python scripts/reconstruct.py jobs confirm --kind baseline > jobs_baseline.txt
scripts/queue_jobs.sh jobs_train.txt 6 gpu                    # GPUs from $GPUS (default "0 1")
scripts/queue_jobs.sh jobs_baseline.txt 3 cpu
python scripts/reconstruct.py confirm confirm                  # prespecified hypothesis tests
python -m pytest tests                                         # derivative and plumbing checks

# 3) manuscript figures and tables (see scripts/manuscript/README.md)
python scripts/manuscript/generate_evidence.py
```

Studies are defined in `configs/reconstruction/`; results go to `report/metrics/reconstruction/`,
trained fields to `models/rev2_*` and job logs to `report/logs/reconstruction/`.

## Running on a Slurm cluster

The study is many small independent jobs (~2.5 GB GPU memory, 10-110 min each), which suits a
Slurm job array. The code arrives by `git clone`; the gitignored inputs are copied once from a
machine that has them:

```bash
# from the local machine: whole-domain cache and study references
tar -cf - data/processed/full report/metrics/reconstruction | ssh sulis "mkdir -p ~/IdealAorta-PINN && tar -xf - -C ~/IdealAorta-PINN"

# on the cluster
bash scripts/sulis/setup.sh                                    # once: conda env "idealaorta"
python scripts/reconstruct.py jobs confirm --kind train > jobs_train.txt
mkdir -p report/logs/reconstruction/slurm
sbatch --array=1-$(grep -c . jobs_train.txt)%40 scripts/sulis/array.slurm jobs_train.txt
squeue -u $USER
```

## Citation

If you use this code, please cite the companion patient-specific study this project extends:

> Ur Rehman et al., *Physics of Fluids* 37(3):031913, 2025.
