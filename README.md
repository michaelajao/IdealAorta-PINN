# IdealAorta-PINN

IdealAorta-PINN reconstructs velocity, relative pressure and wall shear stress from
sparse velocity observations in idealized descending thoracic aortic geometries.
It compares space-time neural fields with linear and radial basis function (RBF)
interpolation, using a shared pressure-integration and near-wall shear calculation.
Neural-field options include data fitting alone, a continuity penalty and momentum
residual penalties.

The supplied configurations use noise-free CFD observations at two instants,
5-10 ms apart. Pressure recovery can use the CFD eddy viscosity as an additional
input. These configurations evaluate reconstruction under known geometry and
boundary conditions rather than a complete imaging acquisition model.

## Installation

Use Python 3.10 or later in a virtual environment, then install the dependencies
from the repository root.

```bash
python -m pip install -r requirements.txt
python main.py --help
```

For GPU training, install a PyTorch build compatible with the available CUDA
environment. Interpolation and analysis commands can run on CPU. Every command
provides its own help through `python main.py <command> --help`.

## Input data

The raw solver exports are not included in this repository. Reconstruction requires
whole-domain ANSYS CFD-Post CSV exports at the expected location.

```text
data/raw/full_2026-09/Case <n>/<time>.csv
```

Each export contains the fluid volume, lumen wall, aneurysm-zone wall, inlet and
outlet blocks. Fields include velocity, gauge pressure, eddy viscosity, turbulent
kinetic energy, specific dissipation rate and the wall-shear vector. Snapshots of
each case must share their mesh and node ordering.

The configured cases use snapshots at 1.775, 1.780, 2.390 and 2.400 s, with
1.788 s replacing 1.775 s for Case 2. Additional exports support the CFD and
structural plots.

| Input | Expected location |
| --- | --- |
| Time-averaged wall shear stress | `data/raw/TAWSS 30.09.2026/` |
| Inner-wall maximum principal stress | `data/raw/MPS/` |
| Cross-sectional speed and turbulent kinetic energy | `data/results_on_slices.csv` |

Convert the whole-domain CSV files into the per-block parquet cache before running
reconstruction.

```bash
python main.py prepare
```

## Running a reconstruction

These commands fit the continuity field and an RBF comparator to the same Case 4
systolic window. Times are supplied in milliseconds and grid spacing in millimeters.

```bash
python main.py train --arm cont --case 4 --times 1775 1780 --target 1780 --grid 2.5 --seed 0 --wphys 0.01
python main.py baseline --method rbf --case 4 --times 1775 1780 --target 1780 --grid 2.5 --seed 0
```

Use `train --help` and `baseline --help` to select other methods, observation
draws, wall-distance exclusions or training settings. Fluid properties are set in
`configs/constants.yaml`.

## Configured studies

Study definitions in `configs/reconstruction/` specify the cases, time windows,
sampling grids and methods for each comparison.

| Configuration | Purpose |
| --- | --- |
| `reference` | Check momentum balance, pressure integration and shear estimation using complete CFD fields |
| `dev_select` | Select physics weights using held-out observations |
| `dev_main` | Compare neural fields and interpolation on Cases 1 and 9 |
| `confirm` | Evaluate the fixed reconstruction settings on Cases 2-8 |

The complete fields of Cases 1-9 informed the shared post-processing checks, so
the confirmation cases are separate from reconstruction development but overlap
with those checks.

Generate job commands and execute them with bounded concurrency.

```bash
python main.py jobs confirm --kind train > jobs_train.txt
python main.py jobs confirm --kind baseline > jobs_baseline.txt
python main.py run-jobs jobs_train.txt --parallel 4 --gpus 0
python main.py run-jobs jobs_baseline.txt --parallel 4 --gpus
python main.py confirm confirm
python main.py report confirm
```

Passing `--gpus` without device IDs selects CPU execution. GPU IDs are assigned
round-robin across concurrent jobs. The `robustness` job kind adds tuned and
space-time RBF comparisons. Run `select dev_select` after weight-selection jobs
and `summarize dev_main` after the development comparison.

## Results

Checkpoints and training histories are written under `models/<run>/`. The
`report/` directory holds reconstruction records in `runs/`, observation indices
in `masks/`, diagnostics in `checks/`, summaries in `tables/`, plots in `figures/`
and execution logs in `logs/`.

Velocity errors are scored on unobserved interior nodes. Pressure errors use a
common gauge and interior nodes more than 1 mm from the wall. Shear errors use
the aneurysm-zone wall nodes. Reported reconstruction errors use the relative
L2 norm.

```bash
python main.py figures cfd maps
```

CFD plots require the corresponding solver exports, while reconstruction maps
also require the fitted fields. Plot generation and study summaries read the
saved run records, so retain those records alongside any checkpoints needed to
reproduce the results.
