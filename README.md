# IdealAorta-PINN

**Parametric physics-informed neural-network (PINN) surrogate for idealized aortic aneurysm hemodynamics.**

This project trains a PINN that learns the hemodynamic field — velocity `(u, v, w)`, pressure `p`,
and wall shear stress (WSS) — as a function of **inlet diameter** and **disease state** for idealized
aortic geometries, and validates it against ANSYS CFD (incompressible RANS with the SST k–ω transition
model). Once trained, the surrogate predicts the flow for a previously unseen inlet diameter
(*leave-one-diameter-out*), and produces 3D streamlines and WSS maps for figures.

It is the *idealized, parametric, generalizing* counterpart to the group's published patient-specific
work (Ur Rehman et al., *Phys. Fluids* 37(3):031913, 2025), and follows the same modelling stack and
PINN "house style" as the `TAA-aneurysm` and `Double_pinn_hemodynamics` projects.

---

## Study design

12 idealized aortic geometries:

| Group      | Inlet Ø (cm) | Asymmetry β = r/R | Notes |
|------------|--------------|-------------------|-------|
| Aneurysmal | 2.0/2.3/2.6  | 1.0 (axisym.), 2.08 (anterior), 0.48 (posterior) | fixed 4.0 cm saccular bulge at mid-vessel |
| Healthy    | 2.0/2.3/2.6  | —                 | smooth linear taper, outlet = 80% inlet |

**Pilot subset** (`representative-first`): axisymmetric diseased Cases 1/4/7 (2.0/2.3/2.6 cm) plus
their healthy controls Cases 10/11/12. Case 4 (2.3 cm) is held out for the headline
leave-one-diameter-out validation.

## Repository layout

```
idealaorta_pinn/            # the Python package
  config.py                 # paths, YAML loaders, physical constants + inlet waveform
  data/                     # CFX parser, case registry, parquet cache, geometry, dataset assembly
  pinn/                     # model (P2INN), physics residual, losses + BCs, trainer
  analysis/                 # inference, validation metrics + WSS, streamlines, figures
configs/                    # YAML: constants, case metadata, per-stage experiments
scripts/                    # prepare.py (data) and run.py (train+validate+figures)
tests/                      # pytest: parser, registry, physics residual
data/                       # raw/ (moved CFD), processed/ (parquet cache), registry.json
models/                     # trained checkpoints (one subfolder per experiment)
report/                     # figures/ (PNG), metrics/ (csv/json/txt), interactive/ (HTML)
paper/                      # LaTeX fragments, references.bib, figures/
```

## Setup

The project uses a dedicated conda env `idealaorta-pinn` (cloned from `dl_env`: Python 3.11,
torch 2.7 + CUDA 12.8, pyvista, vtk, plotly). From the repo root:

```powershell
$py = "C:\Users\ajaoo\miniconda3\envs\idealaorta-pinn\python.exe"
& $py -m pip install -e .          # editable install (also pulls the light deps)
```

Scripts also add the repo root to `sys.path`, so they run without an editable install.

## Pipeline

```powershell
$py = "C:\Users\ajaoo\miniconda3\envs\idealaorta-pinn\python.exe"

# 1) data prep
& $py scripts/prepare.py migrate            # dry-run preview of the move into data/raw
& $py scripts/prepare.py migrate --apply    # move the 12 case folders (+ xlsx) into data/raw
& $py scripts/prepare.py registry           # discover cases -> data/registry.json
& $py scripts/prepare.py cache              # parse CFX CSVs -> data/processed/*.parquet
#    (or: scripts/prepare.py all  -- migrate --apply, registry, cache in one go)

# 2) train + validate + figures (one workflow, since they share the model)
& $py scripts/run.py --config configs/stageA_case1.yaml                 # Stage A de-risk (Case 1)
& $py scripts/run.py --config configs/stageB_richerloo_f16.yaml --cases 4 5 6  # Stage B: predict unseen 2.3 cm (LODO)
& $py scripts/run.py --config configs/stageA_case1.yaml --skip-train    # re-validate/plot an existing model
```

## Running on an HPC (e.g. brosnan, 48 GB GPU)

The heavy runs (Stage C over all 12 cases, longer training, bigger networks) are best on a large GPU.
The project is portable — no scheduler script needed; SSH in (VS Code Remote‑SSH works well) and run
interactively, using `tmux` so long runs survive disconnects.

```bash
# --- on your laptop: publish the repo (once) ---
#   GitHub Desktop -> Publish, or:  git init && git add -A && git commit -m "init" && git push

# --- on brosnan (bash) ---
git clone <your-repo-url> IdealAorta-PINN
cd IdealAorta-PINN

# 1) environment (login node has internet)
conda env create -f environment.yml          # creates env "idealaorta-pinn"
conda activate idealaorta-pinn
# IMPORTANT: install torch matching brosnan's CUDA (check `nvidia-smi`), e.g. CUDA 12.1:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pip install -e .

# 2) data: copy the migrated data from the laptop (run from the LAPTOP)
#   scp -r "C:/Users/ajaoo/Documents/GitHub/IdealAorta-PINN/data/raw"  user@brosnan:~/IdealAorta-PINN/data/
#   (or scp the original CFD export folder and run: python scripts/prepare.py migrate --source <path> --apply)
python scripts/prepare.py registry
python scripts/prepare.py cache               # all 12 cases

# 3) train (use tmux for long runs); 48 GB lets you use the scaled config
tmux new -s ideal
python scripts/run.py --config configs/stageC_hpc.yaml        # full study, 48 GB-scaled
#   detach with Ctrl-b then d;  reattach with: tmux attach -t ideal
```

Outputs land in experiment-scoped folders: `models/<experiment>/` for checkpoints and training
history, `report/{figures,metrics,tables,interactive,logs}/<experiment>/` for generated outputs,
and `paper/figures/<experiment>/` for manuscript figure copies. Use `--device cuda` (default) and,
if you hit memory limits, lower `loaders.max_velocity_points` or `physics.n_collocation` in the
config.

When training an experiment that already has outputs, `scripts/run.py` archives the existing model,
report, and paper-figure folders into sibling `_archive/` directories before starting the new run.
Pass `--overwrite-output` only when you intentionally want to replace outputs in place.

## Method (summary)

- **Inputs:** Fourier-encoded `(x, y, z)` ⊕ a small parameter encoder over `[d_inlet*, β, disease, phase]`
  (P²INN-style parametric conditioning).
- **Networks:** decoupled per-field residual-block Swish MLPs for `u, v, w, p`, plus a smaller
  turbulent-viscosity network `ν_t(x; p)` (softplus-positive). Dual AdamW optimizers (`ν_t` at 10× LR).
- **Physics:** steady RANS-mean momentum + continuity residual with effective viscosity
  `ν_eff = ν_mol + ν_t`, evaluated by autograd in standardized coordinates (quasi-steady per phase).
- **Data fit:** 3D CFD streamline velocity samples + wall pressure/WSS; XY/XZ plane exports are
  reserved for validation and figures. WSS is computed from the velocity gradient at the wall.
- **Loss balancing:** gradient-norm adaptive weighting (Wang et al. 2021) with EMA.
- **Validation:** leave-one-diameter-out vs CFD (relative L2 on velocity, U_ref-normalized NRMSE,
  WSS error, and recirculation). Reference slice CSVs are kept as tables, not plotted as xlsx
  comparisons.

See `outputs/methods_pinn.tex` / `outputs/results_pinn.tex` (generated in Stage C) for manuscript text.

## Reproducibility

Every training run writes checkpoints, the fitted normalizer, loss history, resolved reference
scales, and the random seed under `models/<experiment>/`. Validation outputs are separated by type
under `report/`: figures, metrics, tables, interactive HTML, and logs each live in their own
`<experiment>/` folder.
