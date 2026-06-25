# Project Plan & Status — IdealAorta-PINN

**Single source of truth** for the parametric-PINN aortic-aneurysm surrogate paper:
status, locked decisions, the full experiment campaign, metrics/figures/tables, the
execution order, and the honest limitations. This file consolidates the former
`PLAN.md` (status/roadmap) and `EXPERIMENT_PLAN.md` (campaign matrix) into one
document. Pairs with [`RESEARCH.md`](RESEARCH.md) (methodology + literature),
[`DIAGNOSIS.md`](DIAGNOSIS.md) (the Stage-A debugging trail), and the repo
[`README.md`](../README.md) (setup, pipeline, HPC).

_Last updated 2026-06-23, after the checkpoint-provenance audit caught the
mixed-supervision leakage and the corrected 3D-only re-runs were started._

---

## 0. Critical protocol corrections & provenance — READ FIRST

These four facts override anything stale elsewhere (including `CLAUDE.md`):

1. **Conda env is `deep_tf`, not `idealaorta-pinn`.** The `idealaorta-pinn` env does
   not exist on this machine. Run with its python directly (avoids `conda run`
   swallowing stdout):
   `/home/olarinoyem/miniconda3/envs/deep_tf/bin/python scripts/run.py --config ...`
   Verified: torch 2.6.0+cu124, CUDA on. Hardware: **2× Quadro RTX 8000 (48 GB each)**.
   Pin a job to a card with `CUDA_VISIBLE_DEVICES=N`.

2. **Training protocol is `data.velocity_kinds: ["3D"]` only.** The XY/XZ plane exports
   are *validation and figure* slices, never velocity supervision. XZ is the default
   `--val-kind`; XY the default `--fig-kind`. All config files on disk are now
   corrected to `["3D"]`.

3. **All pre-2026-06-23 checkpoints are leakage-contaminated.** The audit found every
   Stage A/B checkpoint trained before today stored `velocity_kinds:
   ['XY','XZ','3D']` — they trained on the XY/XZ planes AND validated velocity on XZ,
   so their reported velocity errors leak. **Treat all pre-today metrics and the
   numbers currently in `paper/.../results_pinn.tex` as diagnostic only**, not
   paper-grade, until regenerated from corrected runs. Affected & needing fresh
   re-runs: `stageB_richerloo_f16`, `stageB_kfold_hold2p0`, `stageB_kfold_hold2p6`,
   and the old `stageA_case1_s12`.

4. **Re-run fresh, never `--resume`.** An old checkpoint's `resume_state.pt` carries
   the leaky config, so resuming continues the wrong protocol. Launch a fresh run (no
   `--resume`); `run.py` auto-archives the old dir to `models/_archive/<name>_<stamp>`
   (and likewise figures/metrics/interactive/logs). A run is **done** only if
   early-stop fired OR `loss_history.csv` reached `training.epochs` (the convergence
   gate). The headline fold was previously killed at 17000/18000 — do not reuse it.

---

## 1. Status at a glance

**Done**
- Repo, env, data pipeline (`registry → cache → bundle → train → validate → figures`).
  All 12 cases have complete systolic + diastolic 3D/plane/WSS data.
- Locked architecture (see §2). Physics residual MMS-verified to 1e-6
  (`tests/test_mms_physics.py`).
- Metrics implemented: per-phase velocity NRMSE, WSS rel-L2 + peak ratio, recirculation
  fraction, swirl (whole-slice + sac-masked).
- Trainer: dual AdamW, grad-norm adaptive weighting, stationary model-selection monitor,
  batched per-group losses, crash-safe `--resume` (sidecar every `save_interval`).
- **Checkpoint-provenance audit done** (this session): all old checkpoints classified as
  leaky mixed-supervision (§0.3).

**In progress (live runs)** — see §3.
- GPU 0: `stageA_case1_s12` (corrected 3D-only in-sample reconstruction).
- GPU 1: `stageB_richerloo_f16` fresh leave-2.3-out fold (corrected 3D-only).

**Pending — the campaign (see §4, §8)**
- Wave 0 code (metrics B1/C1/C2, table generators, run.py `--seed`/`--name-suffix`,
  leakage-audit + resume-equivalence tests) — CPU-only, do while GPUs are busy.
- Fresh re-runs of the other two LODO folds (2.0-out, 2.6-out) + multi-seed.
- Comparators (physics-OFF, CFD-interp, per-case, NN-diameter).
- Stage C in-sample capacity demo (9-case primary, 12-case stretch).
- Paper rewrite of the PINN section from corrected numbers (§9).

---

## 2. Method recap + locked architecture (do NOT re-litigate)

P²INN-style **parametric** PINN: input `(x,y,z)` Fourier-encoded ⊕ a P2INN parameter
encoder over `μ = [d_inlet*, β, disease_flag, phase]` (n_param=4); **five decoupled**
Swish residual nets `u,v,w,p` + a smaller eddy-viscosity net `ν_t` (softplus + hard
floor). **Single length-scale** nondim (`L`, `U_ref`, `Re = ρU_ref·L/μ ≈ 3.8e4`,
`P_ref = ρU_ref²`, `wss_std`), fit on **train cases only** (no leave-one-out leakage).
Physics = steady RANS-mean momentum + continuity with `ν_eff = 1/Re + ν_t`,
quasi-steady per phase; WSS by autodiff at the wall. Dual AdamW (`ν_t` @ 10× LR),
grad-norm adaptive loss weighting (Wang 2021, EMA + cap, every 250 ep), model selection
on a **stationary monitor** (held-out vel rel-L2 if a holdout exists, else unweighted
component-loss sum) — NOT the adaptive-weighted total. Full rationale in `RESEARCH.md`.

**Locked (established, not up for re-litigation):**
- **S1 per-phase budget equalization ON** — velocity nets emit O(1) for both phases;
  per-phase gain `s = U_ref^phase/U_ref`; all velocity-derived losses divided by `s²`.
  Fixes the diastolic collapse (baseline 44% → single-digit NRMSE; recirc matched
  in-sample).
- **S2 volumetric collocation ON** — rejection-sample the true 3D lumen interior
  (handles the non-convex bulge).
- **Isotropic Fourier, 16 freq (LODO) / 24 freq (Case-1 5-layer).** **S4 (anisotropic
  Fourier) tested and REJECTED** (transverse speckle + peak over-prediction, no WSS
  gain). 12-freq gives best velocity but ~2× worse WSS; WSS is the clinical output, so
  16-freq is retained.
- Dual AdamW + grad-norm adaptive weighting + stationary monitor.

**Discard decision (this session):** stop spending GPU on settled ablations — S4
(`_s124`), low-bandwidth (`_lowbw`), 12-freq (`_f12`). **Keep their configs and
`report/metrics/`** as evidence for the ablation/bandwidth tables; do not delete and do
not re-run. Their model checkpoints are already pruned.

**Dataset (fixed):** 12 idealized cases = 3 inlet diameters (2.0/2.3/2.6 cm) × 3 diseased
symmetry classes (axisym = 1,4,7; anterior = 2,5,8; posterior = 3,6,9) + 3 healthy
(10=2.0, 11=2.3, 12=2.6). **Two phases only** (systolic, diastolic) — not a time series.

---

## 3. Current runs & checkpoint provenance

**Live (2026-06-23):**
| GPU | Run | Config | Protocol | Owner | Notes |
|---|---|---|---|---|---|
| 0 | `stageA_case1_s12` | corrected | 3D-only, S1+S2 | parallel (Copilot) session | in-sample reconstruction ceiling |
| 1 | `stageB_richerloo_f16` | corrected | 3D-only, S1+S2, cases 4/5/6 | this session (PID 635984) | fresh leave-2.3-out (headline); to convergence gate |

**Archived leaky checkpoints** (diagnostic only): `models/_archive/stageB_richerloo_f16_20260623_154335`
and the corresponding figures/metrics/interactive/logs archives; the kfold folds will be
archived likewise when re-run.

**Diagnostic numbers (leaky runs — for reference, NOT for the paper):**
- In-sample Case 1 (old s12): vel NRMSE sys 0.071 / dia 0.049; WSS rel-L2 ~0.02; recirc
  matched (sys 0.048/0.046, dia 0.042/0.043).
- LODO leave-2.3-out (old): mean vel NRMSE ~0.28; WSS rel-L2 0.36–0.54; peak ratio up to
  1.85×; **recirculation over-predicted held-out** (e.g. case5 dia CFD 0.004 / PINN
  0.158; case6 dia 0.000 / 0.120). 2.0-out and 2.6-out folds worse (extrapolation).

---

## 4. Experiment matrix (prioritized P0 / P1 / P2)

Conventions: `(exists)` = runnable config present; `(NEW ← base)` = author a single-purpose
copy of `base` changing only `experiment.name`, `output_dir`, `data.train_cases`,
`random_seed`, point budgets, the relevant toggle, and the header — OR use `run.py`
`--seed`/`--name-suffix` overrides (preferred, Wave 0). **Every run is fresh + corrected
3D-only.** Seed regime: **5 seeds {42,7,123,43,44}** on LODO folds + the in-sample S1+S2
case; **3 seeds {42,43,44}** on clean-ablation rungs; **single seed 42** on heavy/
contingency runs.

| # | Run name | base | toggle / purpose | train → hold | arch / points | seeds | mem | prio |
|---|---|---|---|---|---|---|---|---|
| **Block 1 — clean orthogonal in-sample ablation (add `data.holdout`).** |
| 1 | `stageA_case1` | exists | baseline (S1 off, S2 off) | [1]→[1] | 5L/24f, 40k vel, 6000 coll | 42,43,44 | ~24 GB | P0 |
| 2 | `stageA_case1_s1` | exists | +S1 only | [1]→[1] | 5L/24f | 42,43,44 | ~22 GB | P0 |
| 3 | `stageA_case1_s2` | NEW←_s1 | +S2 only (S1 OFF) | [1]→[1] | 5L/24f | 42,43,44 | ~24 GB | P0 |
| 4 | `stageA_case1_s12` | exists | +S1+S2 | [1]→[1] | 5L/24f | 42,43,44 | ~24 GB | P0 |
| 5 | `stageA_case1_s124` | exists | +S1+S2+S4 (rejected evidence) | [1]→[1] | aniso | 42 | — | P0 (no run) |
| **Block 2 — architecture ablation. On Case 1 μ is near-constant → encoder test must be on LODO.** |
| 6 | `stageA_case1_noparenc` | NEW←_s12 | `use_param_encoder: false` | [1]→[1] | 5L/24f | 42 | ~24 GB | P1 (degenerate) |
| 7 | `stageA_case1_nofourier` | NEW←_s12 | `use_fourier: false` | [1]→[1] | 5L/24f | 42 | ~24 GB | P0 |
| 8 | `stageB_f16_noparenc` | NEW←_richerloo_f16 | encoder off on headline fold | [1,2,3,7,8,9]→[4,5,6] | 6L/16f | 42 | ~35 GB | **P0** (real encoder test) |
| **Block 3 — full k-fold leave-one-diameter-out (6 train / 3 hold, diseased only).** |
| 10 | `stageB_loo_2p0_f16` | NEW←_richerloo_f16 | hold smallest — **extrapolation** | [4,5,6,7,8,9]→[1,2,3] | 6L/16f, 3000 vel/grp, 500 coll/grp | 42,7,123,43,44 | ~35 GB | P0 |
| 11 | `stageB_richerloo_f16` | exists (re-run fresh) | hold middle — **interpolation (headline)** | [1,2,3,7,8,9]→[4,5,6] | 6L/16f | 42,7,123,43,44 | ~35 GB | P0 |
| 12 | `stageB_loo_2p6_f16` | NEW←_richerloo_f16 | hold largest — **extrapolation** | [1,2,3,4,5,6]→[7,8,9] | 6L/16f | 42,7,123,43,44 | ~35 GB | P0 |
| **Block 4 — comparators (justify "physics-informed" + "parametric surrogate"). Headline fold F-2.3.** |
| 13 | `stageB_f16_physoff` | NEW←_richerloo_f16 | `physics: 0`, `physics_floor: 0`, **AND zero/freeze `nut`** | [1,2,3,7,8,9]→[4,5,6] | 6L/16f | 42,7,123 | ~35 GB | P0 |
| 14 | `stageB_f16_percase` | reuse stageA singles | non-parametric per-case floor (no new train) | per-case | — | 42 | — | P0 |
| 15 | CFD-interp baseline | NEW `scripts/baseline_cfd_interp.py` | register 2.0 & 2.6 CFD onto held 2.3 cloud (NN/RBF in normalized coords), blend, score | →[4,5,6] | n/a | n/a | CPU | P0 |
| 16 | NN-diameter baseline | same script `--mode nearest` | use 2.6 field for 2.3 (disclose correspondence caveat) | →[4,5,6] | n/a | n/a | CPU | P1 |
| **Block 5 — Stage C reconstruction (in-sample capacity ONLY, never a generalization claim).** |
| 17 | `stageC_diseased9_s12` | NEW←`stageC_full` | S1+S2, iso 16f, trimmed; **PRIMARY capacity demo** | [1..9]→[1..9] | 6L/16f, 192-w, ~1500 vel/grp, 400 coll/grp, 8k ep | 42 | ~38–42 GB | P1 |
| 18 | `stageC_full_s12` | NEW←`stageC_full` | all 12; **stretch goal** | [1..12]→[1..12] | as above, 24 groups | 42 | ~42–46 GB | P1 (stretch) |
| **Block 6 — healthy / diseased contrast (exercises `disease_flag`).** |
| 19 | `stageHealthy_insample_s12` | NEW←_richerloo_f16 | `train_cases:[10,11,12]` | [10,11,12]→[10,11,12] | 6L/16f | 42 | ~26 GB | P1 |
| 20 | `stageB_loo_healthy23_s12` | NEW←_richerloo_f16 | hold healthy 2.3 | [1..10,12]→[11] | 6L/16f | 42 | ~44 GB | P2 |
| **Block 7 — bandwidth / physics-rigor sweeps. Promote one `_f12` fold to P1 to ground WSS-bandwidth at LODO arch.** |
| 21 | `stageB_loo_2p3_f12` | exists | `num_frequencies: 12` at LODO arch | [1,2,3,7,8,9]→[4,5,6] | 6L/12f | 42 | ~32 GB | **P1** |
| 22 | `stageB_loo_2p0_f12`, `_2p6_f12` | NEW←respective _f16 | 12-freq extrapolation folds | as folds | 6L/12f | 42 | ~32 GB | P2 |
| 23 | `stageB_richerloo_f16_coll{1k,2p5k}` | NEW←_richerloo_f16 | `n_collocation` ∈ {1000,2500} | [1,2,3,7,8,9]→[4,5,6] | 6L/16f | 42 | ~35 GB | P2 |
| 24 | `stageA_case5{,_s1,_s2,_s12}` | NEW←Case-1 set | repeat Block-1 ladder on Case 5 | [5]→[5] | 5L/24f | 42 | ~24 GB | P2 |

**`stageA_case1_s2` (NEW)** = copy `stageA_case1_s1.yaml`, set
`loss_balance.per_phase_velocity_scale: false`, add `loaders.volumetric_collocation: true`.
**P0 count** ≈ Block 1 (4 rungs × 3 seeds = 12) + Block 2 (rungs 7,8) + Block 3 (3 folds ×
5 seeds = 15) + Block 4 (rung 13 × 3 + interp) ≈ **~32 GPU-runs**.

**Reduce config-fork risk:** add `--seed` / `--name-suffix` overrides to `run.py`
(override `random_seed`, `experiment.name`, `output_dir`) + a config-validation assert
(`train_cases ∩ hold = ∅`; `output_dir` matches `name`) rather than forking ~15 YAMLs.

---

## 5. Outputs & metrics per run

`run.py` writes `report/metrics/<exp>/{velocity,wss}.{csv,json,txt}`, figures, interactive
HTML. New metrics flagged `(NEW)` need Wave-0 code. Priority order:
**B1 div > C1 wss-nrmse + C2 Bland-Altman > B2/C3 corr/ccc > B4 region split > D1 wall
pressure > B3/C4 direction > C5 area-frac > E1 streamlines.** Primary error metrics =
per-phase NRMSE (vel) and C1 `wss_nrmse`; demote rel-L2 (diastolic WSS rel-L2 is on a
sub-1-Pa field, not clinically comparable to systolic ~13–17 Pa).

| ID | Metric | Status | File | Note |
|---|---|---|---|---|
| — | vel_rel_l2, vel_nrmse_phase, speed_*, recirc_cfd/pinn, swirl_* | exists | velocity | `velocity_metrics` |
| — | wss_rel_l2, wss_mean_*, wss_peak_* | exists | wss | `wss_metrics` |
| **B1** | div_mean_abs, div_rms, **div_rel = ‖∇·u‖/‖∇u‖_rms** | **NEW** | velocity | new `predicted_divergence()` autograd; frame as "residual continuity" soft constraint; also surface momentum-residual RMS on CFD points (closure proxy) |
| B2 | speed_pearson_r, speed_ccc (+per-component) | **NEW** | velocity | shared `_ccc(a,b)` |
| B3 | vel_cos_mean, vel_angle_p90 (speed-weighted, floored) | **NEW** | velocity | shared `_weighted_cosine()`; reuse recirc floor |
| B4 | vel_nrmse_phase_{bulge,neck}, div_rms_{bulge,neck} | **NEW** | velocity | `_axial_region_mask()` (PCA long-axis, r > 1.15× median) |
| **C1** | **wss_nrmse**, wss_scale (per-phase q0.995 norm) | **NEW** | wss | PRIMARY WSS metric |
| **C2** | wss_ba_bias, _sd, _loa_low/high, _bias_pct | **NEW** | wss | feeds figure B5 |
| C3 | wss_pearson_r, wss_spearman_r, wss_ccc | **NEW** | wss | |
| C4 | wss_cos_mean, wss_angle_p90 | **NEW** | wss | read `wss_x/y/z` |
| C5 | low/high_wss_frac_{cfd,pinn} | **NEW** | wss | state threshold (q0.10/q0.90 or 0.4 Pa) |
| D1 | dp_rel_l2, dp_nrmse, **dp_range_ratio** | **NEW** | new `pressure` | wall-only, mean-removed; `dp_pearson/spearman` PRIMARY; absolute p unconstrained |
| D2 | dp_pearson_r, dp_spearman_r | **NEW** | pressure | |
| E1 | endpoint_rmse, endpoint_p90, n_seeds | **NEW, gated** | new `streamline` | qualitative figure only unless CFD-seeded + fixed integrator + self-consistency baseline |
| F | summary.{csv,json}: mean + worst-case by phase & region | **NEW** | new `summary` | aggregate in `run.py`; surface div_rel, wss_ba_bias, wss_cos_mean in console |

**Statistical aggregation (NEW):** seed-sweep loader globs `report/metrics/<family>_seed*/`
and emits **mean ± std (n)** as primary; t-CI only as a clearly-labeled wide interval.
**No significance tests at n=3** — report raw deltas; frame additivity as "consistent
across 3 seeds." Implemented as `--seeds`/`--aggregate` on `make_ablation_table.py` +
new `scripts/make_seed_table.py`.

**Explicitly NOT computable — state in paper, do not fabricate:** TAWSS/OSI/RRT/transWSS
(2 phases, no ∫dt); absolute/transmural pressure (arbitrary CFX datum, wall-only → report
wall `dp_range_ratio` as a labeled proxy); unsteady residual (quasi-steady by design);
k/ω/ν_t field validation (`nut` is an **effective learned closure**, NOT the SST eddy
viscosity — validate only indirectly via velocity/WSS/divergence + momentum-residual
proxy); CFD mesh-convergence (single mesh — in-sample ~5% is "near characterizable
supervision noise," NOT "near CFD truth"; ship the CFX solver-settings + residual paragraph).

---

## 6. Figure suite

### 6.1 Tier A — per-run diagnostics (auto, `report/figures/<exp>/`, regenerable `--skip-train`)
| ID | File | Status | Claim |
|---|---|---|---|
| A1–A8 | XY/XZ velocity, wss_map, pressure_map, bulge/axial profiles, convergence, streamlines.html | exist | per-(case,phase) QC |
| A9 | `…_residual_map.png` (continuity, momentum-mag, ν_t panels) | **NEW** | field satisfies the PDE; residual concentrates on the jet |
| A10 | `…_scatter_density.png` (speed + WSS hexbin, y=x, slope, R²) | **NEW** | global calibration; jet over-prediction = slope > 1 |
| A11 | `…_error_summary.png` (grouped bars S vs D) | **NEW** | QC card; catches diastolic collapse |
| A12 | `…_bc_check.png` (inlet vel vs Waveform, outlet p) | **NEW** | soft BC terms honored |

### 6.2 Tier B — paper-curated (hand-selected into `paper/figures/`)
| ID | File | Producer | Run | Claim & honesty constraint |
|---|---|---|---|---|
| **B1** | `pinn_schematic.tex/.pdf` | hand TikZ (spec §6.3) | — | the method; "RANS-mean physics + **learned closure**" — must NOT imply transition-SST fidelity |
| B2 | `ablation_insample.pdf` | NEW `ablation_bar()` | Block 1 | S1 removes diastolic collapse, S2 restores WSS, S4 regresses |
| B3 | `loo_field_grid.pdf` (3×3) | NEW `loo_grid()` | `stageB_richerloo_f16` 4,5,6 sys | topology transfers to unseen 2.3 cm; jet over-predicted. Single fold |
| B4 | `error_vs_diameter.pdf` | NEW `error_vs_diameter()` | all 3 folds + in-sample | generalization ≫ in-sample floor; 2.3 interp, 2.0/2.6 extrap; per-fold scales (xref T2); capped by 3-diameter design |
| B5 | `wss_agreement.pdf` (scatter + Bland-Altman) | NEW `wss_agreement()` | `stageB_richerloo_f16` 4,5,6 | WSS over-prediction = positive bias at high WSS; peak up to 1.85× |
| B6 | `streamlines_loo_case04.png` (+ insample case01) | `comparison_figure` + NEW `save_comparison_png` (kaleido) | F-2.3 case4 sys; Case1 sys | 3D structure + recirculation lobe; masked to CFD streamline cloud. Pre-verify kaleido in `deep_tf`, else matplotlib fallback |
| B7 | `insample_case01_systolic_XY.png` (+ diastolic twin) | `plane_comparison` | `stageA_case1_s12` | reconstruction ceiling few-%; diastolic twin shows S1 fix |
| B8 | `convergence_stageB.pdf` (3-panel) | `convergence_curves` + NEW weight panel | `stageB_richerloo_f16` | stable training; selection on stationary monitor. Disclose LODO monitor = unweighted train-loss sum |
| **B9** | `mu_sweep_wss.pdf` | NEW `mu_sweep()` (inference) | any F-2.3 checkpoint | **the parametric claim**: fix geometry+phase, sweep `d*` over [2.0,2.6] incl held 2.3; peak-WSS varies smoothly & brackets held CFD |

### 6.3 B1 TikZ architecture figure — spec
Single-column `tikzpicture` (`tikz`, `arrows.meta`, `positioning`, `fit`). Left→right:
inputs `(x,y,z)` + `μ=[d_inlet*,β,disease,phase]`; encoders (coords → "Random Fourier
features, iso 16-freq @ LODO"; `μ` → "P2INN encoder g_param, dims [32,32]"); concat;
backbone "Swish residual blocks (6L @ LODO)" → five decoupled heads `u,v,w,p` + smaller
`nut` (softplus + floor); physics block "steady RANS-mean momentum + continuity, ν_eff =
1/Re + nut, autograd"; loss block (7 terms) with an explicit `÷s²` brace on the
velocity-derived group (S1); optimization note (dual AdamW · grad-norm adaptive · stationary
monitor). Vector PDF only.

---

## 7. Table list

| ID | Title | Source | Generator | Status |
|---|---|---|---|---|
| T1 | Dataset / case inventory (12) | `registry.json` × `cases.yaml` | NEW `make_dataset_table.py` | new |
| T2 | Non-dim scales per training set | `models/<exp>/normalizer.json` | NEW `make_scales_table.py` | new (1 completed run per train-set) |
| T3 | Network + training hyperparameters | stage YAMLs | hand `.tex` (note 5L/24f vs 6L/16f) | new |
| T4 | In-sample orthogonal ablation (Case 1) | Block-1 JSONs | `make_ablation_table.py --latex` | needs Block-1 runs |
| T4-supp | Bandwidth study (16/12/low-bw, at LODO arch) | `_f12`/`_f16` + Case-1 sweep | same gen, separate `--out` | extend |
| T5 | LODO k-fold generalization | Block-3 JSONs | `make_ablation_table.py --summary mean` + fold/symmetry labels | needs 2 new folds; keep "leave-2.3-out" titling until all land |
| T6 | Seed-sweep robustness (mean ± std, n) | seed-keyed JSONs | NEW `make_seed_table.py` | needs seed runs |
| T7 | Comparator table (which loss terms kept) | Block-4 JSONs + baseline | `make_ablation_table.py` + manual rows | needs Block 4 |

**Three prose numbers to reconcile:** "4.9% diastolic" is **S1+S2** (not S1-alone =
3.9%); `+S1` diastolic WSS regresses (0.112 vs baseline 0.076) → footnote that S1 targets
velocity/recirc, S2 targets WSS; the 0.28/0.38 headline is the **2.3-fold only** → keep
"leave-2.3-out" titling.

**Also required (cheap, high-credibility):** strengthened **leakage-audit test** (assert
`fit_normalizer` bit-identical with/without held case in `records`, `build_bundle` emits
**zero** supervision tensors for any non-train case, held cases enter only via the
validation path); **resume-equivalence test** (resumed run reaches same best monitor
within rel-tol 1e-2, or run that test with `deterministic: true`); ship
`paper/environment.lock`; disclose `deterministic: false` / 5-seed-captures-init+jitter.

---

## 8. Execution order & resource plan (2 shared GPUs, frequent external kills → `--resume`)

Code lands before runs so all runs emit the new metrics/figures in one pass (no
re-validation churn).

**Wave 0 — code, no GPU, DO NOW (GPUs busy with the two live runs):**
1. Metrics **B1, C1, C2** + shared helpers `_ccc`, `_weighted_cosine`; wire into
   `velocity_metrics`/`wss_metrics`. Then B2/C3, B4, C4/C5, D1/D2, summary F.
   Figures A9–A12, B2–B6, B9.
2. `make_ablation_table.py` `--latex`/`--summary`/seed-aggregate; new
   `make_{dataset,scales,seed}_table.py`; `baseline_cfd_interp.py` (with registration).
3. Add `--seed` / `--name-suffix` overrides to `run.py` + `train_cases ∩ hold = ∅`
   assert. Add `data.holdout` to every Block-1/2 Stage-A config. Define physics-OFF to
   zero/freeze `nut`.
4. Add per-epoch wall-clock logging to `train.py` (epochs/min) so budgets are real.
5. Leakage-audit (strengthened) + resume-equivalence tests; `pytest` green.

> **Collision note:** a parallel session may also edit `metrics.py` / `run.py` / configs.
> Coordinate before touching shared files. Safest first slice: B1/C1/C2 in `metrics.py`
> + the leakage-audit test (directly back the headline numbers, lock out the leakage bug).

**Wave 1 — P0 light, pair on cards (~half day):** GPU0 `stageA_case1` → `_s1`
(42,43,44); GPU1 `_s2` → `_s12` (42,43,44) → `nofourier`; CFD-interp baseline on CPU.

**Wave 2 — P0 heavy k-fold + seeds + physics-off:** 15 fold-runs + 3 physics-off +
`stageB_f16_noparenc` ≈ 19 six-case runs split across the two cards, one ~35 GB job per
card, each `--resume`. **Never two 6-case jobs on one card.** Order: seed-42 of all three
folds first (→ T5 + B3/B4/B5/B9), then fill seeds, then physics-off + encoder-off. Budget
~5 days/card (kill-recovery ≥1.5×).

**Wave 3 — P1 (~1–1.5 days):** `stageC_diseased9_s12` (**primary** capacity demo) with a
**50-epoch pre-flight** reading `torch.cuda.max_memory_allocated()` — abort+trim if >40 GB
before committing 8–10 h. `stageHealthy_insample_s12` on the other card; `_f12` LODO fold;
NN-diameter baseline (CPU). `stageC_full_s12` (12-case) is the stretch goal only.

**Wave 4 — P2 (reviewer-driven):** `_f12` extrapolation folds, collocation sweep, Case-5
ablation, 2-case encoder, `stageB_loo_healthy23_s12`.

**Wave 5 — assembly:** build T1–T7 + B1–B9 from completed checkpoints under `--skip-train`;
write the leakage-audit + limitations paragraphs.

**Critical path to a submittable draft:** Wave 0 → Wave 1 (clean ablation = T4/B2) → Wave 2
seed-42 of three folds **to convergence** (T5/B3/B4/B5/B9) + CFD-interp + physics-off
seed-42 (T7). Gate the draft on convergence, not wall-clock.

### Commands
```bash
PY=/home/olarinoyem/miniconda3/envs/deep_tf/bin/python

# data prep (once)
$PY scripts/prepare.py all

# fresh corrected run, pinned to a card (archives any old dir automatically)
CUDA_VISIBLE_DEVICES=1 $PY scripts/run.py --config configs/stageB_richerloo_f16.yaml --cases 4 5 6

# leave-2.0-out / leave-2.6-out folds
CUDA_VISIBLE_DEVICES=0 $PY scripts/run.py --config configs/stageB_kfold_hold2p0.yaml --cases 1 2 3
CUDA_VISIBLE_DEVICES=1 $PY scripts/run.py --config configs/stageB_kfold_hold2p6.yaml --cases 7 8 9

# Stage C (after a 50-epoch memory pre-flight)
CUDA_VISIBLE_DEVICES=0 $PY scripts/run.py --config configs/stageC_hpc.yaml

# re-validate / re-plot a finished checkpoint without retraining
$PY scripts/run.py --config <cfg> --skip-train

# recover a killed run (only for runs already on the corrected config)
$PY scripts/run.py --config <cfg> --resume
```

---

## 9. Recirculation deliverable (answers the co-author's question)

The co-author asked whether the PINN finds recirculation zones. The paper must answer this
directly, from **corrected** runs:
- **Figure:** CFD 3D streamlines vs PINN-traced 3D streamlines for a diseased aneurysm
  (in-sample Case 1 + held-out Case 4), masked to the CFD streamline cloud (figure B6).
- **Table:** CFD/PINN recirculation fraction (whole-slice + sac-masked) and swirl, per
  phase, in T4 (in-sample) and T5 (held-out).
- **Narrative:** the PINN reproduces recirculation zones — in-sample it matches the CFD
  recirculation fraction to within ~0.001 and reproduces the proximal-shoulder lobe in
  location/extent; held-out it places the zone correctly but **over-predicts its extent**
  (state honestly, do not claim quantitative held-out recirculation accuracy).

---

## 10. Honest limitations (state explicitly; do not paper over)

- **3-diameter cap.** Diameter generalization is fundamentally limited by 3 diameters. 2.3
  is interpolation; 2.0-out / 2.6-out are **true extrapolation** with one bracketing
  diameter. Never report one averaged "generalization" number.
- **Per-fold scales differ.** Each fold's Normalizer is fit on its own train-set → report
  only scale-free errors and xref T2.
- **Closure mismatch.** Targets are CFX SST k-ω-transition (URANS-mean); surrogate uses a
  steady RANS-mean residual with a single learned scalar `nut` (no transport). Call `nut`
  an "effective learned closure," validate indirectly, report the momentum-residual proxy.
- **2 phases only** → no TAWSS/OSI/RRT/transWSS.
- **WSS scale.** Diastolic WSS is sub-1-Pa; rel-L2 is dominated by tiny values → primary
  metric is per-phase `wss_nrmse`; always show absolute peak context.
- **Pressure.** Wall-only, arbitrary datum → only gradient/pattern is meaningful.
- **Recirculation** matched in-sample but **over-predicted held-out** — always qualify.
- **CFD reference floor unknown.** Single mesh, no convergence study → in-sample ~5% is
  "near characterizable supervision noise," not "near CFD truth."
- **Statistics.** `deterministic: false` → seeds vary init + cuDNN jitter; report mean ±
  std (n); no significance tests at n=3.
- **Streamline endpoint metric** integrator/seed dependent → qualitative figure only.
- **Param-encoder claim** cannot be tested on a single case (μ constant) → tested on the
  LODO fold (`stageB_f16_noparenc`).
- **Stage C is in-sample capacity only**, never a generalization claim; 9-case primary,
  12-case stretch.
- **Schedule risk.** Shared GPUs are externally killed; `--resume` is load-bearing; budget
  ≥1.5× wall-clock.
- **Paper-tree note.** The PINN paper sources currently sit under `paper/Presentation/`
  (moved there alongside presentation assets in the working tree). Confirm whether that
  move is intentional before building/submitting.

### Files touched (absolute)
- Metrics/figures: `idealaorta_pinn/analysis/{metrics,figures,figures_paper,streamlines}.py`
- Runner/tables: `scripts/run.py`, `scripts/make_ablation_table.py`, new
  `scripts/make_{dataset,scales,seed}_table.py`, new `scripts/baseline_cfd_interp.py`
- Configs: edits of `configs/stage*.yaml` (+ `data.holdout`, physics-off, seed/name CLI)
- Trainer: `idealaorta_pinn/pinn/train.py` (per-epoch timing)
- Tests: `tests/` (leakage-audit strengthened, resume-equivalence; keep `test_mms_physics.py`)
- Paper: `paper/.../pinn_schematic.tex` (new), `methods_pinn.tex`, `results_pinn.tex`,
  `paper/environment.lock` (new)
