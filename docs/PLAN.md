# Project Plan & Status — IdealAorta-PINN

Living handoff document: current status, key decisions, results so far, and the
roadmap with exact commands. Pairs with [`RESEARCH.md`](RESEARCH.md) (methodology
+ literature) and the repo [`README.md`](../README.md) (setup, pipeline, HPC).

_Last updated after Stage A (single-case de-risk) completed._

---

## 1. Status at a glance

**Done**
- [x] Repo scaffolded at `C:\Users\ajaoo\Documents\GitHub\IdealAorta-PINN` (portable; git-ready).
- [x] Dedicated env `idealaorta-pinn` (conda clone of `dl_env`, `opik` removed, `pip install -e .`); torch 2.7 + CUDA, pyvista, vtk, plotly.
- [x] Data migrated into `data/raw/` (12 cases + `Results on Slices.xlsx`, verified); `data/registry.json`; parquet cache in `data/processed/`.
- [x] **Correction:** all 12 cases have complete systolic + diastolic 3D/plane/WSS data (the "missing systolic" was a phase-typo regex artifact, fixed).
- [x] Consolidated package: `config.py` + `data/` + `pinn/` + `analysis/`; **2 scripts** `prepare.py`, `run.py`.
- [x] Tests: 9 passing (parser, registry, physics residual).
- [x] Configs: `stageA_case1` (+ ablations), `stageB_richerloo_f16` & `stageB_kfold_hold2p{0,6}` (LODO k-fold), `stageC_full`, `stageC_hpc` (48 GB).
- [x] Output layout: `models/`, `report/{figures,metrics,interactive}`, `paper/` (LaTeX + `references.bib`); paths anchored to project root.
- [x] Interactive figure = **CFD-traced vs PINN-traced** streamlines (lumen-masked integration), no fallback.
- [x] Trainer model-selection / early-stop on a **stable monitor** (held-out rel-L2 if a holdout exists, else unweighted component-loss sum).
- [x] `docs/RESEARCH.md` (study + literature + methodology), README HPC section.
- [x] **Stage A run completed** (see §3).

**Pending (recommended next — see §4)**
- [ ] Make Stage A an **in-sample reconstruction** de-risk (drop the XZ holdout).
- [ ] Add a **U_ref-normalized velocity metric** (so near-stagnant diastole is meaningful).
- [ ] **Stage B** leave-one-diameter-out (predict 2.3 cm) — the headline result — ideally on **brosnan**.
- [ ] **Stage C** full parametric surrogate (all 12) on brosnan (`configs/stageC_hpc.yaml`).
- [ ] Finalize `paper/methods_pinn.tex` + `paper/results_pinn.tex` once numbers are in.

---

## 2. Method (one-paragraph recap)

P²INN-style **parametric** PINN: input `(x,y,z)` Fourier-encoded ⊕ a parameter
encoder over `μ = [d_inlet*, β, disease, phase]`; decoupled per-field Swish
residual nets `u,v,w,p` + a smaller eddy-viscosity net `ν_t`; **dual AdamW**
(`ν_t` at 10× LR); **gradient-norm adaptive** loss weighting; **single
length-scale** nondimensionalization with a steady RANS-mean residual
`(u·∇)u + ∇p = ∇·[(1/Re + ν_t)(∇u+∇uᵀ)]`; WSS by autodiff at the wall; validation
by **leave-one-diameter-out**. Full rationale + citations in `RESEARCH.md`.

---

## 3. Stage A result (single-case de-risk, Case 1)

Trained on **XY plane + 3D** (XZ plane held out), 637 epochs / 11.8 min, early-stopped.

| Held-out XZ | systolic | diastolic |
|---|---|---|
| velocity rel-L2 | **3.55** (best) | 33.6 (metric artifact) |
| WSS rel-L2 / peak | 0.89 (CFD 24.9 → PINN 6.1 Pa) | 5.9 |

**Findings (incl. inspection of `report/figures/case01_systolic_XY_velocity.png`)**
- **The model is NOT converged.** The figure shows the PINN speed field is **noisy
  and ~2–3 m/s almost everywhere even on the XY plane it was trained on**, versus
  CFD's clean ~0–1.5 m/s field. The "low" training velocity loss (0.10 in
  *standardized* units) is misleading: with `U_ref≈1.42` that is ~0.45 m/s RMSE —
  comparable to the mean flow itself.
- **Root cause = premature early stopping.** The monitor was the held-out
  *orthogonal* XZ plane, which is under-determined and degrades early, so training
  stopped at only **637 epochs** — far too few for a PINN (typically 10k–100k+
  iterations) — before even the training planes fit.
- At **Re≈37,000** the `1/Re` viscous term is tiny, so the physics residual barely
  constrains the solution; with too few epochs the field stays noisy.
- WSS **peak underestimated** (CFD 24.9 → PINN 6.1 Pa; near-wall spectral bias).
- Diastolic rel-L2 (33.6) is a **metric artifact** (near-stagnant; tiny denominator).

**Conclusion:** the infrastructure/pipeline is correct end-to-end, but the PINN
needs **real training**. (a) Run Stage A as **in-sample reconstruction** (drop the
orthogonal holdout so early-stop uses the in-sample loss, not the misleading XZ
plane); (b) train **much longer** (~8k–20k epochs); (c) strengthen physics /
Fourier features / near-wall collocation — ideally on **brosnan**. Stage B
(held-out *diameter*) remains the real generalization test.

---

## 4. Roadmap & commands

Interpreter: `C:\Users\ajaoo\miniconda3\envs\idealaorta-pinn\python.exe` (local) or
`python` inside the activated env on brosnan.

```bash
# data prep (once)
python scripts/prepare.py all                 # migrate --apply + registry + cache

# Stage A — in-sample reconstruction de-risk (after the pending edit: remove holdout,
#           add XZ back to velocity_kinds). Verify low in-sample rel-L2 + falling residual.
python scripts/run.py --config configs/stageA_case1.yaml

# Stage B — leave-one-diameter-out: train on 2.0 + 2.6 cm, predict unseen 2.3 cm (Cases 4-6)
python scripts/run.py --config configs/stageB_richerloo_f16.yaml --cases 4 5 6

# Stage C — full study (all 12), 48 GB-scaled (brosnan; use tmux)
python scripts/run.py --config configs/stageC_hpc.yaml

# re-validate / re-plot any trained model without retraining
python scripts/run.py --config <cfg> --skip-train
```

`run.py` does **train → validate → figures → interactive** in one go. Outputs:
checkpoints in `models/<exp>/`, metrics in `report/metrics/`, PNGs in
`report/figures/` (+ `paper/figures/`), rotatable HTML in `report/interactive/`.

**HPC (brosnan):** see README "Running on an HPC" — clone, `conda env create -f
environment.yml`, install torch matching brosnan's CUDA, `pip install -e .`, scp
`data/raw`, `prepare.py registry && prepare.py cache`, then `run.py` under tmux.

---

## 5. Open decisions / notes for whoever continues

- **Stage A edit not yet applied:** to switch to in-sample reconstruction, in
  `configs/stageA_case1.yaml` set `data.velocity_kinds: [XY, XZ, "3D"]` and remove
  the `data.holdout:` block. (Then the monitor auto-falls back to the unweighted
  component-loss sum.)
- **Train to convergence (key):** 637 epochs was far too few (see §3 — the field
  is noisy even on the training plane). Bump `training.epochs` to ~8k–20k and let
  early stopping (on the in-sample monitor) decide; do this on brosnan.
- **Strengthen physics if needed:** at Re≈37k the viscous term is tiny; consider a
  higher `adaptive_weights.physics_floor`, more `physics.n_collocation`, and more
  `model.num_frequencies` (near-wall structure / WSS peaks).
- **Metric upgrade not yet applied:** add a U_ref-normalized RMSE in
  `analysis/metrics.velocity_metrics` so diastole is interpretable.
- **Stage C memory:** 24 (case,phase) groups; the trainer accumulates their graphs
  before one backward, so memory scales with group count. On OOM, lower
  `loaders.max_velocity_points` / `physics.n_collocation` (documented in the config).
- **WSS peaks:** if peak WSS accuracy matters, raise `num_frequencies`, add
  near-wall collocation banding, and train longer (brosnan).
- **References:** entries in `paper/references.bib` marked UNVERIFIED need confirming
  before submission.
- The original harness plan file under `~/.claude/plans/` is **superseded** by this
  document.
