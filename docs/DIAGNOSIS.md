# Diagnosis & Fix Plan — why the PINN fields are qualitatively wrong

Status: after Stage A (in-sample), Stage B leave-one-diameter-out (3k and 12k epochs), and a
per-group relative-loss ablation. This document is the root-cause analysis and the proposed
solution. **No code has been changed based on it yet** — it is for review before implementation.

Produced from a 13-agent diagnostic (6 root-cause analyses, each adversarially verified against the
code, then synthesized). All file:line citations were confirmed by the verification pass.

---

## 1. What we observe

Converging the leave-one-out run from 3k → 12k epochs improved the headline numbers but did **not**
fix the fields. On held-out **case 4 (2.3 cm, trained on 2.0 + 2.6 cm), 12k epochs (monitor 0.46)**:

| Case 4 (held out) | vel nrmse/U_ref | vel rel-L2 | PINN peak speed | CFD peak | recirc CFD/PINN | WSS peak CFD/PINN |
|---|---|---|---|---|---|---|
| systolic | 13.7% | 0.42 | **2.47 m/s** | 1.3 | 0.071 / 0.104 | 13.6 / 17.2 Pa |
| diastolic | 7.6% | **4.2** | **1.32 m/s** | 0.106 | 0.194 / 0.340 | 0.74 / 1.7 Pa |

In-sample (Stage A, case 1, 12k): systolic nrmse 6.4%, diastolic 2.7% — so **capacity exists**.

**Four symptoms to explain:**
1. **Peak over-prediction in both phases** — systolic 1.9×, diastolic 12×.
2. **Speckled / blotchy fields** — not smooth CFD-like flow.
3. **Diastolic collapse** — the model cannot produce the near-stagnant diastolic field.
4. **Recirculation + WSS systematically over-predicted.**

The global-`U_ref` NRMSE metric *hid* the diastolic collapse (7.6% looks fine; rel-L2 is 4.2).

---

## 2. The real problem

> **The network must represent two flow regimes that differ in magnitude by ~14×, but every scale in
> the pipeline — velocity normalization, Reynolds number, loss budget, and adaptive weighting — is
> pinned to the systolic regime. Simultaneously, the one mechanism that could fill the unsupervised
> 3D interior (the PDE residual) is structurally disabled: it is evaluated only on the two 2D data
> planes it shares with the supervision, and weighted to near-zero.**

This is **not** an architecture/capacity problem (in-sample hits 6.4% / 2.7%). It is a
**scaling-and-constraint problem**. The approach is sound and does not need to be abandoned. A single
relative-loss flip was already shown to over-correct (systolic regressed) — proving the levers are
**coupled** and must move together.

---

## 3. Ranked root causes

All six were *supported* by verification; two dominate (each confirmed from two angles), three are
secondary/contributing.

### DOMINANT

**Cause A — single systolic-scale `U_ref` for both phases.** ([normalize.py:78,95](../idealaorta_pinn/data/normalize.py#L78))
`U_ref=1.404` is the 0.995-quantile of *pooled* speeds; `P_ref=ρU_ref²`, `tau_ref=μU_ref/L`, and `Re`
all derive from it. Diastolic targets become O(0.07) standardized, so absolute MSE
([losses.py:47](../idealaorta_pinn/pinn/losses.py)) gives the diastolic group **~150–200× less
gradient budget** than systolic. The only thing distinguishing the phases is a raw 0/1 scalar through
a shared param-encoder — too weak to switch regimes.
→ Explains **diastolic collapse (3)**, **diastolic peak over-prediction (1)**, **diastolic recirc/WSS (4)**.
*Caveat from verification:* this bites in-sample too (Stage A diastolic rel-L2 1.78); the **systolic**
1.9× overshoot is only weakly from here (systolic is well-supervised) — that's Cause B/C.

**Cause B — the PDE residual has no authority over the interior.** ([loaders.py:196-201](../idealaorta_pinn/data/loaders.py#L196))
Three compounding, code-confirmed facts: (1) collocation points are a **subsample of the velocity
supervision points**, which are ~98% on two near-2D planes (XY z=0, XZ y=0.03) — so the residual is
enforced **only where data already pins the field**, never in the off-plane lumen or bulge core
(no independent volumetric sampler exists); (2) physics is weighted ~0.01; (3) `Re` is fixed at the
systolic scale for both phases, so the standardized momentum balance is mis-scaled ~14× for diastole.
→ Explains **speckle (2)** (unconstrained Fourier ringing off-plane — the real speckle driver),
**systolic peak over-prediction (1)**, contributes to **(4)**.
*Caveat:* the "degenerate/near-constant nut" sub-claim was **refuted** — trained `nu_t` varies strongly
(CV ~15) and advection is quadratic, so the PDE *can* constrain magnitude in principle. The defect is
**coverage + weight + phase-Re mis-scaling**, not a degenerate equation. The full 3D **wall** *is*
constrained (no-slip weight 10), so WSS error is more a near-wall-fit / nut issue than pure coverage.

### SECONDARY / CONTRIBUTING

**Cause C — isotropic Fourier basis on an anisotropic domain.** ([model.py FourierFeatures](../idealaorta_pinn/pinn/model.py))
A single `L` from the *axial* extent makes the standardized domain x:y:z ≈ 2.00:0.71:0.44; an isotropic
`B~N(0,1)` under-resolves the transverse directions where the shear layer/recirculation vary →
**vessel-scale blotch** (too *smooth*, not pixel noise). Shapes the texture of the unconstrained region
but does not create it.

**Cause D — parametric under-sampling (2-diameter LOO).** Trains on only 2.0 + 2.6 cm; held-out 2.3 cm
is the exact normalized midpoint (`d_nd=1.0`), `beta`/`disease_flag` are dead constants. This is the
**in-sample → held-out amplifier** (explains 6.4% → 13.7%), **not** the source of qualitative wrongness
(same defects in-sample). 2-point interpolation yields *intermediate*, not over-the-envelope, values —
so the 1.9× systolic overshoot is **not** an interpolation artifact.

**Cause E — adaptive weighting + metric mask the imbalance.** Adaptive weights key off `ref='velocity'`
over *components* only, never across phases within velocity — so diastole can never be up-weighted.
And [metrics.py:55-56](../idealaorta_pinn/analysis/metrics.py) divides RMSE by the *global* `U_ref`,
so diastolic collapse reads as a benign 7.6%. This is why the failure was hard to see.

---

## 4. Solution plan

**Coupling rule:** a naive single change over-corrects (proven by the phasebal ablation). Land
**S1 + S2 + S3 together** as one coherent change, then S4, then S5. Q1 always; Q2 is an interim bridge.

### Quick wins (tune existing — hours)

- **Q1 — per-phase metrics (do first, always).** [metrics.py](../idealaorta_pinn/analysis/metrics.py).
  Report per-phase rel-L2 and per-phase-`U_ref` NRMSE alongside the global one. No symptom impact, but
  makes every later experiment honest. *1h, no risk.*
- **Q2 — per-phase loss budget (interim, NOT relative-only).** [losses.py](../idealaorta_pinn/pinn/losses.py)
  + train loop. Keep absolute MSE but add a per-(case,phase) weight so the diastolic group's *effective*
  gradient budget matches systolic (≈ `0.5*absolute + 0.5*relative`). The relative-only flip over-corrected;
  a balanced budget gets diastole without wrecking systole. *2–3h, medium risk — ship only with Q1.*
- **Q3 — raise physics weight + densify collocation — ONLY after S2.** Configs: `physics_floor 0.01→~0.1`,
  `n_collocation 2500→6000+`. High risk if done before S2 (stronger enforcement of a plane-only residual
  makes things worse, and re-introduces the 48 GB OOM). *Do not ship standalone.*

### Structural changes (the real fixes)

- **S1 — per-phase velocity scaling (most important).** `normalize.py`, `loaders.py` data path,
  `predict.py`, `metrics.py`. Fit each phase's own 0.995-quantile so both phases' standardized targets
  are O(1). **Critical trap:** `P_ref`, `tau_ref`, the inlet BC, and `Re` all derive from `U_ref` and
  the residual assumes one scale — so **keep ONE geometry-driven `U_ref`/`L`/`Re` for the physics
  residual, pressure, and WSS, and apply per-phase scaling ONLY to the velocity *data-loss target***
  (rescale `u_t/v_t/w_t` per phase; rescale predictions back before physics/BC terms). Do **not** make
  the whole Normalizer phase-dependent — that breaks the momentum derivation.
  *Impact:* should largely eliminate **(3)** (diastolic peak 1.32 → ~0.1–0.2; rel-L2 6 → <1) and the
  diastolic half of **(4)**. *~1–2 days; old checkpoints become incompatible (update Normalizer
  to/from_dict, re-fit). S1 and Q2 are alternative routes to equal phase budget — prefer S1; don't stack.*
- **S2 — independent volumetric collocation sampler (second most important).**
  [loaders.py:196-201](../idealaorta_pinn/data/loaders.py#L196). Replace `rng.choice` over velocity
  points with **uniform rejection sampling inside the lumen**, using the wall cloud + inward normals
  (already at loaders.py:207) for a signed-distance interior mask. The only fully-supported cause of the
  unconstrained interior. *Impact:* large on **(2)**, meaningful on **(1)** systolic. *2–3 days; HIGH
  risk — the bulge is non-convex, a convex-hull/tube mask leaks into the wall; validate the signed
  distance via the wall KD-tree before trusting it.*
- **S3 — phase-dependent Reynolds number in the residual.** `train.py` / `compute_physics_loss`
  (data already groups by phase). The standardized diastolic momentum balance is currently mis-scaled
  ~14×; no reweighting fixes that. *1 day; design together with S1's single-scale-for-physics decision
  so velocity-data scaling and residual scaling stay self-consistent.*
- **S4 — anisotropic Fourier basis.** `model.py` FourierFeatures: scale `B` per-axis by inverse
  standardized span (x:1.0, y:~2.84, z:~4.53). Use the **per-axis Fourier-scale** route, NOT per-axis
  `L` (which touches the physics Laplacian/Re). *Secondary on (2); 0.5 day; over-scaling transverse can
  reintroduce near-wall noise.*
- **S5 — stronger phase conditioning + parametric coverage (defer).** Small learned phase embedding or
  per-phase output gain; pull asymmetric/healthy cases into training to populate the diameter×beta
  manifold. *Amplifier, not root cause — defer until A+B are fixed; Stage C is GPU-bound (OOM risk).*

---

## 5. What to test first

**Highest-leverage experiment: S1 (per-phase velocity scaling, physics kept single-scale) + Q1
(per-phase metrics), re-validated leave-one-out on case 4, watching BOTH phases jointly.**
Lowest-effort structural fix, attacks the most-confirmed dominant cause, targets the worst symptom (12×).

**Success looks like:**
- Diastolic peak speed `1.32 → ~0.1–0.2 m/s` (CFD 0.106); diastolic rel-L2 `4.2 → <1`.
- **Systolic does not regress** (rel-L2 ≤ 0.45, recirc near 0.10) — the guardrail the relative-loss flip failed.
- Diastolic recirc `0.34 → ~0.2`, WSS peak `1.7 → ~0.8 Pa`.

**Honest caveat:** S1 alone will **not** fix the systolic 1.9× overshoot or the speckle — those live in
Cause B/C (S2/S3/S4). If systolic is still blotchy after S1, that *confirms* the diagnosis and says
proceed to S2+S3. If diastole does *not* improve under S1, the scaling thesis is wrong → escalate to
the phase-embedding rethink (S5) first.

**Discipline (the verifiers' demand):** before claiming any held-out win, re-run **Stage A in-sample**
with vs without each structural change. If Stage A barely moves, that change targets the LOO amplifier
(Cause D), not the root defect — redirect effort. This distinguishes the dominant causes (which move
in-sample numbers too) from the secondary amplifier.

**Bottom line:** the approach can work. It needs (1) correct per-phase non-dimensionalization and
(2) a physics residual actually allowed to constrain the 3D interior. Fix those two; the
parametric/Fourier items are real but secondary polish.
