# IdealAorta-PINN — Unified Experiment Plan

Single source of truth for the experiment campaign behind the parametric-PINN aortic-aneurysm
surrogate paper. Merges five expert designs (experiments, metrics, figures, tables, rigor) and
folds in every actionable item from the adversarial review. Conflicts are resolved inline. The
locked architecture is **not** up for re-litigation: five decoupled nets (`u,v,w,p` + `nut`),
random Fourier coordinate features, P2INN parameter encoder, single-length-scale
nondimensionalization, **S1 on, S2 on, isotropic Fourier, S4 off**, dual AdamW (`nut` @ 10× lr),
gradient-norm adaptive loss weighting (Wang 2021), and a **stable** model-selection monitor (NOT
the adaptive-weighted total).

Throughput/runtime numbers below are planning estimates anchored on one externally-killed 6-case
run; they MUST be re-derived from the first instrumented completed run (see Wave 0, item 5, and
review Gap 16) before the full matrix is committed.

---

## 1. Overview & what is already established

### 1.1 The method (one paragraph)
One model conditioned on `mu = [d_inlet*, beta, disease_flag, phase]` predicts velocity `(u,v,w)`,
pressure `p`, and wall shear stress (WSS). Five decoupled scalar nets (`u,v,w,p` + a smaller
turbulent-viscosity `nut`), each Fourier-encoded coords + P2INN parameter encoder + Swish residual
blocks. Single-length-scale nondim (`L`, `U_ref`, `Re = rho*U_ref*L/mu`, `P_ref = rho*U_ref^2`,
`wss_std`), fit on **train cases only**. Physics is the steady RANS-mean momentum + continuity
residual with `nu_eff = 1/Re + nut`, quasi-steady per phase. Trained with dual AdamW + grad-norm
adaptive weighting; model selection on a stationary monitor.

### 1.2 Established this week — do NOT re-run or re-litigate
- **S1 per-phase budget equalization**: velocity nets emit O(1) for both phases; per-phase gain
  `s`; all velocity-derived losses divided by `s^2`. Fixes diastolic collapse (baseline diastolic
  44% NRMSE down to single digits; recirculation matched in-sample).
- **S2 volumetric collocation**: rejection-sample the true 3D lumen interior (handles the
  non-convex bulge).
- **Isotropic Fourier, 16 frequencies** for LODO (WSS is the clinical output). **S4 (anisotropic
  Fourier) was tested and rejected** (transverse-velocity speckle + peak over-prediction, no WSS
  gain). 12-freq gives best velocity but ~2× worse WSS.
- Trainer: batched per-group losses (vectorized segment-mean), one-graph `train_step` (fixed
  adaptive-update OOM), crash-safe `--resume` (resume sidecar written every `save_interval`,
  default 1000 epochs — confirmed at `train.py:550`).
- Physics residual MMS-verified to 1e-6 (`tests/test_mms_physics.py`).
- Metrics implemented: per-phase NRMSE, WSS rel-L2, WSS peak ratio, recirculation fraction (M4).

### 1.3 Dataset (fixed)
12 idealized cases = 3 inlet diameters (2.0, 2.3, 2.6 cm) × 3 diseased symmetry classes
(axisymmetric = 1,4,7; anterior = 2,5,8; posterior = 3,6,9) + 3 healthy (10 = 2.0, 11 = 2.3,
12 = 2.6). **Two phases only** (systolic, diastolic) — NOT a time series.

### 1.4 Known gaps these experiments close (and the review items keyed to each)
- (a) Ablation not orthogonal (S1+S2 validated together; `+S1` row pre-C1) → **Block 1**.
- (b) No seed sweep → **Block 3/Block 1 multi-seed**, review Gaps 3, 20, 26.
- (c) Only 2.3 cm held out → **Block 3 full k-fold**, review Gaps 4, 7.
- (d) Diameter generalization capped by 3 diameters → state honestly everywhere (review Gap 4, 22).

### 1.5 Must-fix gates BEFORE any number enters a table or the paper (review P0)
1. **Stage-A holdout claim is false in configs** (Gap 1). VERIFIED: only `stageA_case1.yaml`
   sets `data.holdout`; `_s1` and `_s12` do not, so `train.py` falls back to the unweighted
   training-loss sum (the same regime as LODO). **Fix:** add a `data.holdout` block (one in-sample
   plane never trained on) to every Block-1/Block-2 Stage-A config so selection is honest and
   matches `methods_pinn.tex`; reconcile CLAUDE.md + methods text + disclosure note. Do not ship
   the contradiction.
2. **Headline LODO run under-trained** (Gap 2). VERIFIED: `stageB_richerloo_f16/loss_history.csv`
   has 17001 rows vs an 18000 cap; no early-stop fired; it was externally killed, monitor still
   descending. **Fix:** a run is "done" only if early-stop fired OR `loss_history` reached `epochs`.
   Re-run all P0 LODO folds to that gate before any number ships. `--resume` is load-bearing.
3. **`deterministic` set in no config** → defaults `False` (`train.py:58`); cuDNN nondeterministic.
   Seed sweep then conflates init variance with kernel jitter. **Fix:** (a) resume-equivalence
   test asserts *approximate* equality (rel-tol 1e-2) OR runs once with `deterministic: true`;
   (b) disclose seeds vary init+kernel jitter; (c) document one regime for headline folds.
4. **Per-fold scales not common** (Gap 4). The Normalizer is fit per train-set, so F-2.0/F-2.3/F-2.6
   use different `L/U_ref/Re`. Report only scale-free errors (per-phase NRMSE normalizes by the
   held case's own speed — confirm WSS C1 NRMSE is likewise scale-free) and cross-reference T2 in
   the B4 caption. **2.3 = interpolation; 2.0-out and 2.6-out = true extrapolation** (one bracketing
   diameter only) — never average into one "generalization" number.
5. **Ablation arch ≠ LODO arch** (Gap 7). Block-1 Case-1 configs are 5L/24-freq/encoder[16,16];
   LODO is 6L/16-freq/encoder[32,32]. Either run the Block-1 ladder at the LODO backbone OR scope
   every ablation claim to "Case-1 5-layer backbone" AND re-derive the 16-vs-12 WSS trade at the
   LODO arch (promote a `_f12` fold to P1).
6. **Comparators under-specified** (Gaps 9, 10): physics-OFF must also zero/freeze `nut`;
   CFD-interp is geometrically ill-posed (non-matching meshes) and needs real registration code,
   not "~min CPU".
7. **Mixed-epoch seed-42** (Gap 26): the killed 17000-epoch F-2.3 seed-42 must be re-run to the
   completion gate for parity before it sits in a CI with 18000-epoch seeds.

---

## 2. Experiment matrix (prioritized P0 / P1 / P2)

Conventions: `(exists)` = already runnable; `(NEW ← base)` = author a single-purpose copy of `base`,
changing only `experiment.name`, `output_dir`, `data.train_cases`, `random_seed`, point budgets,
the relevant toggle, and the header. Mem budgets are peak including the adaptive-update spike.
Runtimes are to the `epochs` cap; early stop (patience 2000) usually trims 20–40% — but the kill
budget (review Gap 14) means plan ~1.5× wall-clock for recovery. **Seed regime (conflict resolved):
5 seeds {42,7,123,43,44} on LODO folds and the in-sample S1+S2 case (load-bearing CI claims),
3 seeds {42,43,44} on the clean-ablation rungs, single seed 42 on heavy/contingency runs.**

| # | Run name | base config | toggle / purpose | train → hold (case ids) | arch / points | seeds | GPU·mem | runtime | prio |
|---|---|---|---|---|---|---|---|---|---|---|
| **Block 1 — clean orthogonal in-sample ablation (closes gap a). Add `data.holdout` per Gap 1. Decide arch per Gap 7.** |
| 1 | `stageA_case1` | (exists) | baseline (S1 off, S2 off) | [1] → [1] | 5L/24-freq, 40k vel, **6000** coll | 42,43,44 | 1·~24 GB | ~5–6 h ea | P0 |
| 2 | `stageA_case1_s1` | (exists) | +S1 only | [1] → [1] | 5L/24-freq, 40k vel, 4000 coll | 42,43,44 | 1·~22 GB | ~5–6 h ea | P0 |
| 3 | `stageA_case1_s2` | NEW ← `_s1` | +S2 only (**S1 OFF**) | [1] → [1] | 5L/24-freq, 40k vel, 4000 coll | 42,43,44 | 1·~24 GB | ~5–6 h ea | P0 |
| 4 | `stageA_case1_s12` | (exists) | +S1+S2 | [1] → [1] | 5L/24-freq, 40k vel, 4000 coll | 42,43,44 | 1·~24 GB | ~7–8 h ea | P0 |
| 5 | `stageA_case1_s124` | (exists, done) | +S1+S2+S4 (rejected evidence) | [1] → [1] | aniso, 40k vel, 4000 coll | 42 | — | done | P0 (no run) |
| **Block 2 — architecture ablation (closes Gap 7/R7). On Case 1 mu is near-constant → repeat on [1,7] or run on LODO (Gap 12).** |
| 6 | `stageA_case1_noparenc` | NEW ← `_s12` | `model.use_param_encoder: false` | [1] → [1] | 5L/24-freq | 42 | 1·~24 GB | ~7–8 h | P1 (degenerate; see Gap 12) |
| 7 | `stageA_case1_nofourier` | NEW ← `_s12` | `model.use_fourier: false` | [1] → [1] | 5L/24-freq | 42 | 1·~24 GB | ~7–8 h | P0 |
| 8 | `stageB_f16_noparenc` | NEW ← `_richerloo_f16` | `use_param_encoder: false` on the headline fold | [1,2,3,7,8,9] → [4,5,6] | 6L/16-freq | 42 | 1·~35 GB | ~8–9 h | **P0** (real encoder test, Gap 12) |
| 9 | `stageA_set17_parenc{,_off}` | NEW ← `_s12` | encoder on/off, 2-case so mu varies | [1,7] → [1,7] | 5L/24-freq | 42 | 1·~28 GB | ~7 h ea | P2 |
| **Block 3 — full k-fold leave-one-diameter-out (closes gap c; 5 seeds → gap b). 6 train / 3 hold, diseased only.** |
| 10 | `stageB_loo_2p0_f16` | NEW ← `_richerloo_f16` | hold smallest diameter — **extrapolation** | [4,5,6,7,8,9] → [1,2,3] | 6L/16-freq, 3000 vel/grp, 500 coll/grp | 42,7,123,43,44 | 1·~35 GB | ~8–9 h ea | P0 |
| 11 | `stageB_richerloo_f16` | (exists, =seed42, **UNDER-TRAINED — re-run, Gap 2/26**) | hold middle — **interpolation (headline)** | [1,2,3,7,8,9] → [4,5,6] | 6L/16-freq, 3000 vel/grp, 500 coll/grp | 42,7,123,43,44 | 1·~35 GB | ~8–9 h ea | P0 |
| 12 | `stageB_loo_2p6_f16` | NEW ← `_richerloo_f16` | hold largest diameter — **extrapolation** | [1,2,3,4,5,6] → [7,8,9] | 6L/16-freq, 3000 vel/grp, 500 coll/grp | 42,7,123,43,44 | 1·~35 GB | ~8–9 h ea | P0 |
| **Block 4 — comparators / baselines (closes R5/R6; justifies "physics-informed" + "parametric surrogate"). Headline fold F-2.3.** |
| 13 | `stageB_f16_physoff` | NEW ← `_richerloo_f16` | `loss_weights.physics: 0`, `adaptive_weights.physics_floor: 0`, **AND zero/freeze `nut`** (Gap 9) | [1,2,3,7,8,9] → [4,5,6] | 6L/16-freq | 42,7,123 | 1·~35 GB | ~8–9 h ea | P0 |
| 14 | `stageB_f16_percase` | reuse `stageA` single-diameter runs | non-parametric per-case floor (no new train) | per-case | — | 42 | — | reuse | P0 |
| 15 | CFD-interp baseline | analysis only — NEW `scripts/baseline_cfd_interp.py` | **register 2.0 & 2.6 CFD onto held 2.3 point cloud (NN/RBF in normalized coords after scale-align), then blend, score** (Gap 10) | — → [4,5,6] | n/a | n/a | CPU | real impl+validate, **hours not min** | P0 |
| 16 | NN-diameter baseline | same script, `--mode nearest` | use 2.6 cm field for 2.3 (same correspondence caveat, disclose) | — → [4,5,6] | n/a | n/a | CPU | ~min | P1 |
| **Block 5 — Stage C reconstruction (in-sample capacity demo ONLY, never a generalization claim). Primary = 9-case (Gap 15).** |
| 17 | `stageC_diseased9_s12` | NEW ← `stageC_full` | S1+S2, iso, 16-freq, trimmed; **PRIMARY capacity demo** | [1..9] → [1..9] | 6L/16-freq, 192-w, ~1500 vel/grp, 400 coll/grp, 8k ep | 42 | 1 GPU ·~38–42 GB | ~8–10 h | P1 |
| 18 | `stageC_full_s12` | NEW ← `stageC_full` | all 12; **stretch goal** | [1..12] → [1..12] | as above, 24 groups | 42 | 1 GPU ·~42–46 GB | ~9–12 h | P1 (stretch) |
| **Block 6 — healthy / diseased contrast (exercises `disease_flag`).** |
| 19 | `stageHealthy_insample_s12` | NEW ← `_richerloo_f16` | `train_cases:[10,11,12]` | [10,11,12] → [10,11,12] | 6L/16-freq, 8k vel/grp, 1500 coll/grp, 15k ep | 42 | 1·~26 GB | ~5–6 h | P1 |
| 20 | `stageB_loo_healthy23_s12` | NEW ← `_richerloo_f16` | hold healthy 2.3 | [1..10,12] → [11] | 6L/16-freq, 1300 vel/grp, 400 coll/grp | 42 | 1 GPU ·~44 GB | ~10–12 h | P2 |
| **Block 7 — bandwidth / physics-rigor sweeps (R8/R10). Promote one `_f12` fold to P1 to ground the WSS-bandwidth claim at LODO arch (Gap 7).** |
| 21 | `stageB_loo_2p3_f12` | ← `stageB_richerloo_f12` (exists) | `num_frequencies: 12` at LODO arch | [1,2,3,7,8,9] → [4,5,6] | 6L/12-freq, ~4000 vel/grp | 42 | 1·~32 GB | ~8 h | **P1** (Gap 7) |
| 22 | `stageB_loo_2p0_f12`, `stageB_loo_2p6_f12` | NEW ← respective `_f16` | `num_frequencies: 12` | as folds | 6L/12-freq | 42 | 1·~32 GB | ~8 h ea | P2 |
| 23 | `stageB_richerloo_f16_coll{1k,2p5k}` | NEW ← `_richerloo_f16` | `n_collocation` ∈ {1000, 2500} | [1,2,3,7,8,9] → [4,5,6] | 6L/16-freq | 42 | 1·~35 GB | ~8–9 h ea | P2 |
| 24 | `stageA_case5{,_s1,_s2,_s12}` | NEW ← Case-1 set | repeat Block-1 ladder on Case 5 | [5] → [5] | 5L/24-freq | 42 | 1·~24 GB | ~6 h ea | P2 |

**`stageA_case1_s2` (NEW)** = copy of `stageA_case1_s1.yaml`, set
`loss_balance.per_phase_velocity_scale: false`, add `loaders.volumetric_collocation: true`, rename.
**Counts:** P0 ≈ Block 1 (rungs 1–4 × 3 seeds = 12 new) + Block 2 (rungs 7,8 = 2) + Block 3
(3 folds × 5 seeds − 1 reusable-but-must-rerun = 15 runs) + Block 4 (rung 13 × 3 seeds + interp
analysis) ≈ **32 GPU-runs**. P1 ≈ Block 2 rung 6 + Stage-C 9-case + healthy in-sample + `_f12`
LODO + NN baseline. P2 ≈ ~10 (Stage-C 12, healthy-out, `_f12` extrap folds, collocation sweep,
Case-5 ladder, 2-case encoder).

**Reduce config-fork risk (Gap 17):** prefer adding `--seed` / `--name-suffix` CLI overrides to
`run.py` (override `random_seed`, `experiment.name`, `output_dir`) over forking ~15 seed YAMLs,
plus a config-validation assert (`train_cases ∩ hold = ∅`; `output_dir` matches `name`).

---

## 3. Outputs & metrics per run

Every `run.py` already writes `report/metrics/<exp>/{velocity,wss}.{csv,json,txt}`, figures, and
interactive HTML. New metrics are flagged `(NEW)` and require code (Wave 0). Priority order
(rigor-aligned): **B1 div > C1 wss-nrmse + C2 Bland-Altman > B2/C3 corr/ccc > B4 region split >
D1 wall pressure > B3/C4 direction > C5 area-frac > E1 streamlines.** Make per-phase NRMSE (vel)
and C1 `wss_nrmse` the **primary** error metrics; demote rel-L2 (Gap 11: diastolic WSS rel-L2 is on
a sub-1-Pa field and not clinically comparable to systolic ~13–17 Pa).

| ID | Metric | Status | File | Implementation note |
|---|---|---|---|---|
| — | vel_rel_l2, vel_nrmse_phase, speed_*, recirc_cfd/pinn | exists | velocity | `velocity_metrics` |
| — | wss_rel_l2, wss_mean_*, wss_peak_* | exists | wss | `wss_metrics` |
| **B1** | div_mean_abs, div_rms, **div_rel = ‖∇·u‖/‖∇u‖_rms** | **NEW** | velocity | new `predicted_divergence()` mirroring `predict_wss_physical` autograd; report `div_rel` (dimensionless, frame as "residual continuity", soft constraint — Gap 18); also surface momentum-residual RMS on CFD points as the closure proxy (Gap 5) |
| B2 | speed_pearson_r, speed_ccc (+per-component) | **NEW** | velocity | shared `_ccc(a,b)` helper |
| B3 | vel_cos_mean, vel_angle_p90 (speed-weighted, floored) | **NEW** | velocity | shared `_weighted_cosine()`; reuse recirc floor |
| B4 | vel_nrmse_phase_{bulge,neck}, div_rms_{bulge,neck} | **NEW** | velocity | new `_axial_region_mask(coords)` (PCA long-axis, radius > 1.15× median) |
| **C1** | **wss_nrmse**, wss_scale (per-phase q0.995 norm) | **NEW** | wss | PRIMARY WSS metric (Gap 11) |
| **C2** | wss_ba_bias, _sd, _loa_low/high, _bias_pct | **NEW** | wss | feeds figure B5 |
| C3 | wss_pearson_r, wss_spearman_r, wss_ccc | **NEW** | wss | |
| C4 | wss_cos_mean, wss_angle_p90 | **NEW** | wss | read `wss_x/y/z` (confirmed present in all 24 parquet files) |
| C5 | low/high_wss_frac_{cfd,pinn} | **NEW** | wss | state threshold (q0.10/q0.90 or 0.4 Pa) |
| D1 | dp_rel_l2, dp_nrmse, **dp_range_ratio = robust (q0.99−q0.01)** | **NEW** | **new `pressure`** | wall-only, mean-removed; `dp_pearson/spearman` is PRIMARY (pattern), range is secondary; absolute p unconstrained (Gap 19) |
| D2 | dp_pearson_r, dp_spearman_r | **NEW** | pressure | wall `p` confirmed present |
| E1 | endpoint_rmse, endpoint_p90, n_seeds | **NEW, gated** | **new `streamline`** | behind `--streamline-metrics`; **qualitative figure only — do NOT table unless CFD seed points + fixed integrator + self-consistency baseline** (Gap 13) |
| F | summary.{csv,json}: mean + worst-case of vel_nrmse_phase, wss_nrmse, div_rel, wss_ba_bias_pct, by phase & region | **NEW** | **new `summary`** | aggregate in `run.py`; surface div_rel, wss_ba_bias, wss_cos_mean in console |

**Statistical aggregation (NEW post-processing):** seed-sweep loader globs
`report/metrics/<family>_seed*/` and emits **mean ± std (n)** as primary; t-CI only as a clearly
labeled wide interval, never tight (Gap 20: n=5 t-CI barely better than std, n=3 nearly
meaningless). **No significance tests at n=3** — report raw deltas/effect sizes; frame additivity
as "consistent across 3 seeds," not "significant." Implemented as `--seeds`/`--aggregate` on
`make_ablation_table.py` plus new `scripts/make_seed_table.py`.

**Explicitly NOT computable — state in paper, do not fabricate:** TAWSS/OSI/RRT/transWSS (2 phases,
no ∫dt); absolute/transmural pressure (arbitrary CFX datum, wall-only → report wall `dp_range_ratio`
as a labeled proxy only); unsteady residual (quasi-steady by design); k/ω/ν_t field validation (not
exported — `nut` is an **effective learned closure**, NOT the SST eddy viscosity (Gap 5); validated
only indirectly via velocity/WSS/divergence + the momentum-residual proxy); CFD mesh-convergence
(single mesh, out of scope — so in-sample ~5% is "near the supervision noise we can characterize,"
NOT "near CFD truth"; ship the CFX solver-settings + residual-convergence paragraph, Gap 8).

---

## 4. Figure suite

### 4.1 Tier A — per-run diagnostics (auto, `report/figures/<exp>/`, regenerable under `--skip-train`)

| ID | File | Status | Claim |
|---|---|---|---|
| A1–A8 | XY/XZ velocity, wss_map, pressure_map, bulge/axial profiles, convergence, streamlines.html | exist | per-(case,phase) reconstruction QC |
| A9 | `case{ID}_{phase}_residual_map.png` (continuity, momentum-mag, ν_t panels) | **NEW** `physics_residual_map()` | field satisfies the PDE; residual concentrates on the centerline jet; surface its RMS as the closure proxy (Gap 5) |
| A10 | `case{ID}_{phase}_scatter_density.png` (speed + WSS hexbin, y=x, slope, R²) | **NEW** | global calibration; jet over-prediction = slope > 1 |
| A11 | `case{ID}_error_summary.png` (grouped bars S vs D) | **NEW** | at-a-glance QC card; catches diastolic collapse |
| A12 | `case{ID}_{phase}_bc_check.png` (inlet vel vs Waveform, outlet p) | **NEW** | soft BC terms honored |

### 4.2 Tier B — paper-curated (hand-selected into `paper/figures/`)

| ID | File | Producer | Run / data | Claim & honesty constraint |
|---|---|---|---|---|
| **B1** | `pinn_schematic.tex/.pdf` | hand-authored **TikZ** (spec below) | — | the method; "RANS-mean physics + **learned closure**" — must NOT imply fidelity to transition-SST (Gap 5) |
| B2 | `ablation_insample.pdf` | **NEW** `ablation_bar(csv)` | Block 1 (Case 1) | S1 removes diastolic collapse, S2 restores WSS, S4 regresses. Caption: `+S1` row is post-C1 clean run (gap a). "4.9% diastolic" is **S1+S2**, not S1-alone (= 3.9%) |
| B3 | `loo_field_grid.pdf` (3×3) | **NEW** `loo_grid()` | `stageB_richerloo_f16` cases 4,5,6 systolic | bulk topology transfers to unseen 2.3 cm across all 3 symmetry classes; jet core over-predicted. Caption: single fold |
| B4 | `error_vs_diameter.pdf` | **NEW** `error_vs_diameter()` | all 3 folds + in-sample | generalization >> in-sample floor. **Caption: 3 diameters; 2.3 interpolation, 2.0/2.6 extrapolation; each fold uses its own train-set scales (xref T2); plot per-(fold, symmetry), matched-symmetry in-sample floor, capped by design** (Gaps 4, 22) |
| B5 | `wss_agreement.pdf` (scatter + Bland-Altman) | **NEW** `wss_agreement()` | `stageB_richerloo_f16` 4,5,6 | WSS over-prediction = positive bias at high WSS; peak up to 1.85×. Single fold |
| B6 | `streamlines_loo_case04.png` (+ `_insample_case01`) | `comparison_figure` + **NEW** `save_comparison_png` (kaleido) | F-2.3 case 4 systolic; Case 1 systolic | 3D structure + recirculation lobe. Caption: masked to CFD streamline cloud. **Pre-verify kaleido renders in `deep_tf`; fallback to matplotlib 3D quiver or HTML screenshot** (Gap 25) |
| B7 | `insample_case01_systolic_XY.png` (+ diastolic twin) | `plane_comparison` (already wired) | `stageA_case1_s12` | reconstruction ceiling few-%; diastolic twin shows S1 fixes the low-speed phase |
| B8 | `convergence_stageB.pdf` (3-panel: losses, monitor, adaptive weights) | `convergence_curves` + **NEW** weight panel | `stageB_richerloo_f16` | stable training; selection on stationary monitor not weighted total. **Caption must disclose LODO has no held-out holdout → monitor = unweighted training-loss sum** (Gap 1) |
| **B9** | `mu_sweep_wss.pdf` | **NEW** `mu_sweep()` (inference only) | any completed F-2.3 checkpoint | **the parametric claim**: fix geometry+phase, sweep `d*` over [2.0,2.6] incl. held 2.3; predicted peak-WSS / centerline-speed vary smoothly and bracket the held CFD point. Nearly free; directly supports "parametric" (Gap 21) |

### 4.3 B1 TikZ architecture figure — spec
Single-column `tikzpicture` (`\usepackage{tikz}`, `arrows.meta`, `positioning`, `fit`). Left to
right:
1. **Inputs**: a coords node `(x,y,z)` and a parameter node `mu = [d_inlet*, beta, disease_flag,
   phase]` (n_param = 4).
2. **Encoders**: coords → "Random Fourier features (iso, 16 freq @ LODO)"; `mu` → "P2INN parameter
   encoder `g_param(mu)`, dims [32,32]". Concatenate (only coords are Fourier-encoded; params index
   a geometry family).
3. **Backbone**: a stack labeled "Swish residual blocks (6L @ LODO)" feeding **five decoupled
   heads**: `u, v, w, p` (`FieldNet`) and a smaller `nut` (`NutNet`, softplus-positive + hard
   floor), drawn slightly apart to signal decoupling.
4. **Physics block**: a brace from `(u,v,w,p,nut)` into "Steady RANS-mean momentum + continuity,
   `nu_eff = 1/Re + nut`, quasi-steady per phase, autograd derivatives."
5. **Loss block**: list the 7 terms (data-vel, data-p, WSS, no-slip, physics-residual, inlet-vel
   BC, outlet-p BC) with an explicit **`÷ s²` brace** on the velocity-derived group (S1).
6. **Optimization annotation**: "Dual AdamW (flow nets + `nut` @ 10× lr) · grad-norm adaptive
   weighting (Wang 2021) · model selection on stationary monitor."
Use a muted palette; keep one accent color for the S1 `÷s²` brace and one for the physics block.
Export to `pinn_schematic.pdf` (vector). No raster.

---

## 5. Table list

| ID | Title | Columns | Data source | Generator | Status |
|---|---|---|---|---|---|
| T1 | Dataset / case inventory (12 cases) | Case, d(cm), d*, Health, Symmetry, β, [Sac(cm) — **only if derivable from registry/geometry, else drop**, Gap 23], Phases | `data/registry.json` × `configs/cases.yaml` | **NEW** `scripts/make_dataset_table.py` | new |
| T2 | Non-dim scales per training set | TrainSet, L, U_ref, U_ref^dia, Re, P_ref, τ_ref, wss_std | `models/<exp>/normalizer.json` | **NEW** `scripts/make_scales_table.py` | new (needs 1 completed run per train-set → order after Wave-2 seed-42) |
| T3 | Network + training hyperparameters | Group, Hyperparameter, Value (Stage-A vs Stage-B columns; **note 5L/24f vs 6L/16f split**, Gap 7) | stage YAMLs | hand-authored `.tex` | new |
| T4 | In-sample orthogonal ablation (Case 1) | Treatment, Phase, Vel-NRMSE, **WSS-NRMSE (primary)**, WSS-relL2, Peak ratio, Recirc CFD/PINN | Block-1 JSONs | `make_ablation_table.py` + `--latex` | extend; **Block 1 runs needed** |
| T4-supp | Bandwidth study (separate) | 16/12/low-bw rows, **at LODO arch** | `_f12`/`_f16` folds + Case-1 sweep | same generator, separate `--out` | extend (Gap 7) |
| T5 | LODO k-fold generalization | Held-diam, **Nature (interp/extrap)**, Case(symmetry), Phase, Vel-NRMSE, WSS-NRMSE, Peak, **Recirc CFD/PINN per phase** (Gap 6), + per-fold mean | Block-3 JSONs | `make_ablation_table.py` + `--summary mean` + symmetry/fold labels from registry | extend; **2 new folds needed**; keep titled "leave-2.3-out" until all folds land |
| T6 | Seed-sweep robustness | Regime, Phase, Vel-NRMSE, WSS-NRMSE, Peak, Recirc (**mean ± std, n**; wide t-CI only if shown) | seed-keyed JSONs | **NEW** `scripts/make_seed_table.py` | new; **seed runs needed** (Gap 20) |
| T7 | Comparator table | Method (PINN / physics-OFF / CFD-interp / NN-diam / per-case), **which loss terms kept**, Phase, Vel-NRMSE, WSS-NRMSE | Block-4 JSONs + baseline script | `make_ablation_table.py` rows + manual baseline rows | extend; **Block 4 needed** (Gap 9 — state exactly which terms each comparator keeps) |

**Three prose numbers to reconcile:** "4.9% diastolic" is **S1+S2** (not S1-alone = 3.9%);
`+S1` diastolic WSS regresses (0.112 vs baseline 0.076) → footnote that S1 targets velocity/recirc,
S2 targets WSS; the 0.28/0.38 headline is the **2.3-fold only** → keep "leave-2.3-out" titling.

**Generator enhancements to `make_ablation_table.py`:** `--latex` (booktabs; peak as ratio;
recirc as `a / b` cell), `--summary mean` (per-fold/overall mean row), `--seeds`/aggregate
(mean ± std), symmetry/fold-label columns from `data/registry.json`.

**Also required (cheap, high-credibility):**
- **Leakage-audit regression test** — necessary AND sufficient version (Gap 27): assert
  `fit_normalizer` is bit-identical with/without the held case in `records`, AND that `build_bundle`
  emits **zero supervision tensors** for any case not in `train_cases`, AND that held cases enter
  only via the separate validation path. (The actual leakage vector is `data.train_cases`, not just
  the normalizer.)
- **Resume-equivalence check** — assert the resumed run reaches the same best monitor **within
  rel-tol 1e-2** (or run that single test with `deterministic: true`), since cuDNN is nondet by
  default (Gap 3).
- Ship `paper/environment.lock` (torch 2.6 + cu124, CUDA, driver, GPU). Disclose the
  `deterministic: false` / 5-seed-captures-init+kernel-jitter choice.

---

## 6. Execution order & resource plan (2 shared GPUs, frequent external kills → every multi-case run launched with `--resume`)

Code lands before runs so all runs emit the new metrics/figures in one pass (no re-validation churn).

**Wave 0 — code, no GPU, do first:**
1. Metrics **B1, C1, C2** (highest priority) + shared helpers `_ccc`, `_weighted_cosine`; wire into
   `velocity_metrics`/`wss_metrics`. Then B2/C3, B4, C4/C5, D1/D2, summary F. Figures A9–A12, B2–B6,
   B9.
2. `make_ablation_table.py` `--latex`/`--summary`/seed-aggregate; new
   `scripts/make_{dataset,scales,seed}_table.py`; `scripts/baseline_cfd_interp.py` (with the
   registration step, Gap 10).
3. Author the P0 NEW configs (`stageA_case1_s2`, `stageA_case1_nofourier`, `stageB_f16_noparenc`,
   `stageB_loo_2p0_f16`, `stageB_loo_2p6_f16`, `stageB_f16_physoff`) **OR** add `--seed` /
   `--name-suffix` CLI overrides to `run.py` + a `train_cases ∩ hold = ∅` validation assert (Gap 17,
   preferred). **Add `data.holdout` to every Block-1/2 Stage-A config (Gap 1).** Define physics-OFF
   to also zero/freeze `nut` (Gap 9).
4. **Add per-epoch wall-clock logging to `train.py`** (epochs/min over last N) so budgets are real
   (Gap 16) — the 37 ep/min anchor is unverified.
5. Leakage-audit (strengthened, Gap 27) + resume-equivalence (approx, Gap 3) tests; run
   `pytest tests/test_mms_physics.py` + new tests green.

**Wave 1 — P0 light, pair on cards (~half day):**
- GPU0: `stageA_case1` → `stageA_case1_s1` (seeds 42,43,44) sequential.
- GPU1: `stageA_case1_s2` → `stageA_case1_s12` (seeds 42,43,44) sequential; then `nofourier`.
- CFD-interp baseline (Block 4) on CPU anytime here.

**Wave 2 — P0 heavy k-fold + seeds + physics-off (one ~35 GB job per card, sequential per card;
re-derive wall-clock from the first instrumented run before committing):** 15 fold-runs + 3
physics-off + `stageB_f16_noparenc` ≈ 19 six-case runs split across the two cards. **Never two
6-case jobs on one card.** Each `--resume`. **Order within:** seed-42 of all three folds first
(headline T5 + B3/B4/B5/B9 immediately) — **re-run F-2.3 seed-42 to the completion gate (Gap 2/26),
do not reuse the 17000-epoch checkpoint** — then fill seeds, then physics-off + encoder-off. Budget
~5 days/card (kill-recovery ≥1.5×, Gap 14), NOT 3.5. A run counts as done only if early-stop fired
OR `loss_history` hit `epochs` (convergence gate, Gap 2).

**Wave 3 — P1 (~1–1.5 days):** `stageC_diseased9_s12` (**primary** capacity demo) with a
**50-epoch pre-flight** reading `torch.cuda.max_memory_allocated()` — abort+trim if > 40 GB before
committing 8–10 h (Gap 15); do NOT assume solo occupancy on a shared box.
`stageHealthy_insample_s12` on the other card (~6 h); `_f12` LODO fold (Block 7 rung 21, P1);
NN-diameter baseline (CPU). `stageC_full_s12` (12-case) is the stretch goal only.

**Wave 4 — P2 (reviewer-driven only):** `_f12` extrapolation folds, collocation sweep, Case-5
ablation, 2-case encoder, `stageB_loo_healthy23_s12`.

**Wave 5 — assembly:** build T1–T7 + B1–B9 from completed checkpoints under `--skip-train`; write
the leakage-audit + limitations paragraphs.

**Critical path to a submittable draft:** Wave 0 → Wave 1 (clean ablation = T4/B2) → Wave 2 seed-42
of three folds **to convergence** (T5/B3/B4/B5/B9) + CFD-interp + physics-off seed-42 (T7). That set
alone supports the headline claims; remaining seeds and P1/P2 deepen statistics and breadth. **Gate
the draft on convergence (Gap 2), not wall-clock.**

---

## 7. Open robustness items & honest limitations

State these explicitly in the paper; do not paper over them.
- **3-diameter cap.** Diameter generalization is fundamentally limited by 3 diameters. 2.3 is
  interpolation; 2.0-out and 2.6-out are **true extrapolation** with a single bracketing diameter
  on one side. Never report one averaged "generalization" number (Gap 4).
- **Per-fold scales differ.** Each fold's Normalizer is fit on its own train-set, so the three folds
  are not on a common nondimensionalization; report only scale-free errors and xref T2 (Gap 4, 18).
- **Closure mismatch.** Targets are CFX SST k-ω-transition (unsteady-RANS-mean); the surrogate uses
  a steady RANS-mean residual with a single learned scalar `nut` (no transport). Call `nut` an
  "effective learned closure," validate it only indirectly (velocity/WSS/divergence + the
  momentum-residual proxy), and report that proxy RMS on CFD points (Gap 5).
- **2 phases only.** No TAWSS/OSI/RRT/transWSS (no ∫dt). State this constraint plainly.
- **WSS scale.** Diastolic WSS is a sub-1-Pa field; its rel-L2 is dominated by tiny values. Primary
  metric is per-phase `wss_nrmse`; always show absolute peak context (Gap 11).
- **Pressure.** Wall-only, arbitrary CFX datum → only the gradient/pattern is meaningful; report
  `dp_pearson/spearman` as primary and robust range ratio as secondary (Gap 19).
- **Recirculation.** Matched in-sample but **over-predicted in held-out diastole** (e.g. case 5
  diastolic recirc 0.004 CFD vs 0.158 PINN). Always qualify the "recirculation matched" claim
  (Gap 6).
- **CFD reference floor unknown.** Single mesh, no mesh-convergence study; in-sample ~5% is "near
  the characterizable supervision noise," not "near CFD truth." Ship the CFX solver-settings +
  residual-convergence disclosure (Gap 8).
- **Statistics.** `deterministic: false` by default → seeds vary init + cuDNN kernel jitter; report
  mean ± std (n); no significance tests at n=3 (Gap 3, 20).
- **Streamline endpoint metric** is integrator/seed dependent → qualitative figure only unless
  CFD-seeded with a fixed integrator + self-consistency baseline (Gap 13).
- **Param-encoder claim** cannot be tested on a single case (mu constant) → tested on the LODO fold
  (`stageB_f16_noparenc`, P0) and optionally a 2-case set, not on Case 1 alone (Gap 12).
- **Stage C is in-sample capacity only**, never a generalization claim; 9-case primary, 12-case
  stretch (Gap 15).
- **Schedule risk.** Shared GPUs are externally killed (one headline run already died at 17000/18000);
  `--resume` is load-bearing (sidecar every 1000 epochs); budget ≥1.5× wall-clock (Gap 2, 14).

### Files touched (absolute)
- Metrics/figures: `/home/olarinoyem/IdealAorta-PINN/idealaorta_pinn/analysis/metrics.py`,
  `.../analysis/figures.py`, `.../analysis/figures_paper.py`, `.../analysis/streamlines.py`.
- Runner/tables: `/home/olarinoyem/IdealAorta-PINN/scripts/run.py`,
  `.../scripts/make_ablation_table.py`, new `scripts/make_{dataset,scales,seed}_table.py`,
  new `scripts/baseline_cfd_interp.py`.
- Configs: copies/edits of `/home/olarinoyem/IdealAorta-PINN/configs/stageA_case1_s1.yaml`,
  `stageA_case1_s12.yaml`, `stageB_richerloo_f16.yaml`, `stageC_full.yaml` (+ `data.holdout`,
  physics-off, seed/name CLI overrides in `run.py`).
- Trainer: `/home/olarinoyem/IdealAorta-PINN/idealaorta_pinn/pinn/train.py` (per-epoch timing).
- Tests: `/home/olarinoyem/IdealAorta-PINN/tests/` (leakage-audit strengthened, resume-equivalence;
  keep `test_mms_physics.py`).
- Paper: `/home/olarinoyem/IdealAorta-PINN/paper/figures/pinn_schematic.tex` (new),
  `paper/methods_pinn.tex` (B1, leakage + closure paragraphs, Stage-A monitor correction),
  `paper/results_pinn.tex` (B2–B9, T4–T7), `paper/environment.lock` (new).
