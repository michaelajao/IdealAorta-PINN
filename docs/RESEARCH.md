# Research & Methodology Notes — IdealAorta-PINN

This document records the background research, the literature that informs the
method, the design decisions and their justifications, and the data findings for
the parametric PINN-surrogate contribution to the paper *"Hemodynamic and
Morphometric Analysis of Aortic Models under Aneurysmal and Non-Aneurysmal
Conditions."* References are collected in [`paper/references.bib`](../paper/references.bib).

---

## 1. Study overview

**Goal.** Build a *parametric* physics-informed neural-network (PINN) surrogate
that learns the hemodynamic field — velocity `(u,v,w)`, pressure `p`, and wall
shear stress (WSS) — as a function of **inlet diameter** and **disease/asymmetry
state** for idealized aortic geometries, validated against the project's ANSYS
CFD (incompressible RANS, SST k–ω **transition** model), and producing **3D
streamlines** and WSS as outputs.

**Why this is novel for the group.** The collaborator's prior papers fit
*per-case* PINNs (one training run per geometry). The new contribution is a
*single* network conditioned on the geometry parameters that **generalizes to an
unseen inlet diameter** (leave-one-diameter-out), which is the natural next step
their own work points to. The collaborator's explicit request was to "reproduce
3D streamlines in the aneurysm cases," which we deliver by tracing streamlines
through the learned field.

**Design (12 idealized geometries).**

| Group      | Inlet Ø (cm) | Asymmetry β = r/R | Notes |
|------------|--------------|-------------------|-------|
| Aneurysmal | 2.0/2.3/2.6  | 1.0 (axisym.), 2.08 (anterior/"upper"), 0.48 (posterior/"lower") | fixed 4.0 cm saccular bulge at mid-vessel |
| Healthy    | 2.0/2.3/2.6  | —                 | smooth linear taper, outlet = 80% inlet |

**Pilot subset (representative-first):** axisymmetric diseased Cases 1/4/7
(2.0/2.3/2.6 cm) + healthy controls 10/11/12. Case 4 (2.3 cm) is the
leave-one-diameter-out target.

---

## 2. Data

Per case, ANSYS CFX exports (two snapshots: systolic peak ~1.78 s, diastolic
2.4 s):

- **3D streamline traces** — sparse points with `x,y,z` + `u,v,w` + speed.
- **Dense plane velocity fields** — XY plane (z≈0) and XZ plane (y≈0.03),
  ~80k–320k points each.
- **Wall WSS + pressure clouds** — surface points with pressure and WSS vector.

Physical setup (from the manuscript): blood treated Newtonian (ρ≈1060 kg/m³,
μ≈0.0035 Pa·s); pulsatile inlet `Q_max = 2.39e-4 m³/s`, `T = 0.8 s`; outlet
pressure 80–120 mmHg. Peak inlet speeds ≈ 0.8–1.6 m/s; peak Reynolds number
(diameter-based) ≈ 3500–4600 — **transitional**, which is why the CFD used the
transition-SST model.

**Key correction (important).** An early read suggested Cases 4–9 lacked the
*systolic* 3D streamline file. This was an **artifact** of a filename-typo regex
not catching the `Sysstolic` / `Syastolic` spellings. After fixing
`normalize_phase`, **all 12 cases have complete systolic + diastolic 3D / plane /
WSS data**. The dataset is fully complete.

---

## 3. Prior work — the group's "house style"

From the user's three prior repositories (`Double_pinn_hemodynamics`,
`Anuerysm_transientFlow_PINNs`, `TAA-aneurysm`) and publications:

- **Published predecessor:** Ur Rehman, …, Ajao-Olarinoye, … *Physics of Fluids*
  37(3):031913 (2025), DOI 10.1063/5.0259296 — patient-specific Marfan aneurysms,
  CFD (Newtonian + SST k–ω transition) + FSI + a PINN surrogate for WSS. Same CFD
  stack as this study.
- **In press:** Fatima, …, Ajao-Olarinoye, *Pramana* (2026) — ResNet-PINNs (the
  `Double_pinn_hemodynamics` work).
- **In preparation:** Ur Rehman, …, Ajao-Olarinoye, … — thoracoabdominal aneurysm
  FSI + PINN surrogate (the `TAA-aneurysm` work; most advanced, shares the
  idealized axisymmetric/anterior/posterior taxonomy).

**House conventions (mirrored here):** hand-rolled **PyTorch** (no DeepXDE/JAX);
Fourier-feature coordinate encoding; **Swish** activation; residual blocks;
**decoupled per-field subnetworks** (the meaning of "double/multi-PINN" — an
ensemble coupled only through the shared physics loss); a learnable
**eddy-viscosity field ν_t** trained by a separate optimizer at higher LR;
**gradient-norm adaptive loss weighting** (Wang 2021); data-driven
nondimensionalization fit on training cases only; WSS computed by autodiff at the
wall; YAML-driven configs.

---

## 4. Literature review

### 4.1 Sparse-data PINN lineage (Arzani)
- **Arzani, Wang, D'Souza, *Phys. Fluids* 33:071905 (2021)** — "Uncovering
  near-wall blood flow from sparse data with PINNs." The canonical method for
  recovering velocity/WSS from sparse measurements by enforcing the NS residual,
  with WSS obtained by autodiff at the wall. **Its stated future-work gap —
  parametric geometry generalization and pulsatile flow — is exactly the niche we
  occupy.** (The user referred to this as "Asani" → confirmed Arzani.)
- Raissi, Yazdani, Karniadakis, *Science* (2020) — "Hidden fluid mechanics":
  conceptual origin of NS-constrained reconstruction of velocity + pressure from
  limited data.
- Arzani & Dawson (2021), Arzani et al. ABME (2022): motivation for
  physics-leaning data-driven cardiovascular surrogates. Arzani BL-PINN JCP (2023):
  near-wall/boundary-layer treatment if WSS gradients are stiff.

### 4.2 Parametric-input PINNs (the core of our method)
- **P²INN — Cho et al., ICML/PMLR 235 (2024).** The solution is
  `u_Θ(x,t;μ) = g_θg([ g_θc(x,t) ; g_θp(μ) ])`: a coordinate encoder `g_θc`, a
  *separate* parameter encoder `g_θp` (FC stack → higher-dimensional latent), and
  a manifold network `g_θg`. Key empirical finding: **explicitly encoding the PDE
  parameters into a hidden representation** beats treating them "merely as a
  coordinate" (their ablation, PINN-P). Their 2D Helmholtz benchmark **trains on
  parameter `a ∈ [2.5,3.0]` and predicts the unseen `a = 2.75`** — i.e.
  interpolation in parameter space, *the same protocol as our
  leave-one-diameter-out*.
- **IP-PINN — Pashaei Kalajahi, …, Arzani, D'Souza, *Eng. Appl. AI* (2025).**
  Parameterizes a PINN by an *input field* (a 4D-Flow MRI ROI) via a **U-Net flow
  encoder → latent vector**, then `Θ₂(L, x*,z*,t*) → (u,v,w,p)`, generalizing to
  unseen inputs without retraining. Nondimensionalization `x*=(x−x_min)/L`,
  `u*=u/U`, `p*=p/(ρU²)`; loss `(1−α)·data + α·physics`.

**Why P²INN, not IP-PINN, for us.** IP-PINN's conditioning input is a *field/
function* (an image), which warrants a U-Net encoder. Our conditioning is a few
*scalars* (inlet diameter, β, disease, phase) that fully determine an idealized
geometry — so the **P²INN scalar parameter-encoder** is the correct, lighter
analog. (A geometry/point-cloud encoder, IP-PINN/GAPINN-style, is the natural
swap for a future **patient-specific** extension, where geometry is no longer
reducible to scalars.) DeepONet/operator learning was considered and rejected:
it needs many samples and a function-valued input.

### 4.3 Parametric / operator surrogates (context)
Sun et al. CMAME (2020) — physics-constrained parametric surrogate, small-N
feasibility; Oldenburg GAPINN (2022) — geometry-aware conditioning; Lu et al.
DeepONet (2021) — the operator alternative (better when the parameter is a
function and N is large). Cruz-González et al. (2025, arXiv:2503.17402) — closest
published twin (idealized AAA, ANSYS, PINN vs DeepONet) but parameterizes inlet
*velocity*, steady & laminar.

### 4.4 Turbulence / RANS-PINN
Eivazi et al. *Phys. Fluids* (2022) — RANS-PINN with Reynolds stresses as outputs;
Hanrahan et al. *Phys. Rev. Fluids* (2023) — eddy-viscosity-augmented mean-flow
reconstruction; Pioch et al. *Fluids* (2023) — which RANS closure to embed. These
justify our learnable effective-viscosity treatment instead of re-solving the
SST k–ω transport equations (we lack the exported k, ω fields).

### 4.5 PINN training methodology
Raissi et al. JCP (2019, base method + Adam→L-BFGS); Wang et al. SISC (2021,
gradient-norm adaptive weighting — our weighting scheme); Tancik et al. NeurIPS
(2020, Fourier features for near-wall gradients); McClenny & Braga-Neto JCP (2023,
self-adaptive per-point weights); Wang et al. (2023, "Expert's Guide" — modified
MLP, curriculum); Sukumar & Srivastava CMAME (2022, exact/hard BC via distance
functions); Krishnapriyan et al. NeurIPS (2021, failure modes / curriculum).

---

## 5. Methodology (chosen design + justification)

**Inputs / conditioning (P²INN-style).** Network input is `(x,y,z)` →
**Fourier-feature** encoding (coordinate encoder; addresses spectral bias for
sharp near-wall gradients, Tancik 2020) concatenated with a **parameter encoder**
over `μ = [d_inlet*, β, disease_flag, phase]`. Raw concatenation ("PINN-P") is
available as an ablation (`use_param_encoder=False`). β is included so the
asymmetric cases (which share `(diameter, disease, phase)` with the axisymmetric
ones) are distinguishable.

**Architecture.** Decoupled per-field residual-block **Swish** networks for
`u,v,w,p`, plus a smaller **eddy-viscosity network ν_t(x;μ)** (softplus-positive,
hard floor). **Dual AdamW optimizers** — the flow networks on all losses, the ν_t
network on the physics loss at ~10× LR — so the closure field receives dedicated
signal (TAA pattern).

**Physics (per-phase quasi-steady).** A **single length scale** `L` maps physical
to standardized coordinates (`x = x_mean + L·x_s`), with `u = U_ref·u_s` and gauge
`p = ρU_ref²·p_s` (cf. IP-PINN Eq. 17). The dimensionless steady RANS-mean
residual then has every derivative in the *same* standardized coordinate and a
single effective coefficient:

```
(u_s·∇_s)u_s + ∇_s p_s = ∇_s·[ (1/Re + ν_t_s)(∇_s u_s + ∇_s u_sᵀ) ],   ∇_s·u_s = 0
```

with `Re = ρ U_ref L / μ` and `ν_t_s` the non-dimensional eddy viscosity. This is
dimensionally consistent and avoids ad-hoc coordinate-scale bookkeeping. The two
CFD snapshots are treated as two **quasi-steady** reconstructions conditioned on
the `phase` input (we cannot resolve the time derivative from two snapshots, and
state so).

**Turbulence honesty.** We fit the RANS-*mean* field with an effective viscosity
`ν_eff = 1/Re + ν_t` rather than re-solving SST k–ω. We report ν_t magnitude and
intend laminar/constant-ν_t ablations: if ν_t is everywhere small, that
empirically justifies a near-laminar approximation at this Re. We never silently
fit laminar NS to RANS-mean data.

**Losses.** Data fidelity (dense plane velocities + sparse 3D velocities + wall
pressure + WSS) + PDE residual at interior collocation points + soft no-slip +
inlet/outlet BCs from the waveform. WSS is computed by **autodiff** of the
velocity gradient at the wall (Newtonian: standardized WSS = standardized strain;
physical = ×`τ_ref = μU_ref/L`). Loss terms are balanced by **gradient-norm
adaptive weighting** with EMA (manual weights as fallback).

**Nondimensionalization.** Fit `x_mean, L, U_ref, P_ref, τ_ref, Re, D_ref,
wss_std` on the **training cases only**; reused for held-out cases → no leakage in
leave-one-diameter-out.

**Optimizer schedule.** AdamW (flow 1e-4, ν_t 10×) + CosineAnnealing; gradient
clip 1.0; early stopping on training loss; (L-BFGS polish optional).

**Validation.** Leave-one-diameter-out: train on 2 diameters, predict the unseen
one, compare to CFD. Metrics: relative L2 on velocity (held-out plane), WSS
magnitude error (mean/peak), recirculation fraction agreement, and a slice-table
cross-check against `Results on Slices.xlsx`. Stages: **A** = Case 1 de-risk (hold
out the XZ plane); **B** = leave-2.3 cm-out (train 1 & 7, predict 4) — the
headline result; **C** = all 12 (adds healthy + asymmetric via β).

---

## 6. Novelty positioning

The strongest honest claim: *a single parameterized-input PINN, conditioned on
inlet diameter + disease/asymmetry, for idealized aortic saccular aneurysms under
pulsatile transition-SST CFD, validated by leave-one-diameter-out, outputting 3D
streamlines and WSS.* Differentiators:

- **vs the group's PoF 2025** — that is patient-specific, per-case, no parametric
  generalization; ours is the idealized, parametric, *generalizing* counterpart on
  the same CFD stack.
- **vs the TAA in-prep work** — that fits per-geometry networks; ours is a single
  meta-network with held-out-diameter validation (its own stated future work).
- **vs Arzani 2021** — steady, single-geometry, laminar; we are parametric and
  quasi-steady over systolic/diastolic snapshots.
- **vs Cruz-González 2025** — they parameterize inlet *velocity*, steady &
  laminar; we parameterize anatomical *diameter + disease*, pulsatile transitional,
  with LOO.

---

## 7. Risks & mitigations

1. **Only two snapshots** → frame as quasi-steady reconstructions conditioned on
   phase; no temporal-dynamics claim.
2. **Spectral bias / steady-collapse** in the systolic sac vortex → Fourier
   features, sac up-weighting, curriculum.
3. **Laminar-vs-RANS honesty** → learnable ν_t field + report its magnitude +
   laminar ablation.
4. **Ill-conditioned multi-term loss** → gradient-norm adaptive weighting; hard BC
   option to remove BC-weight tuning.
5. **Geometry / wall-normal noise → WSS error** → Open3D/PCA normals, near-wall
   collocation banding.
6. **Small N (12) / extrapolation** → physics carries generalization; report
   interpolation vs extrapolation separately; disclose out-of-range extrapolation
   as unverified.

---

## 8. References

See [`paper/references.bib`](../paper/references.bib). Entries marked UNVERIFIED
(in-press / in-prep DOIs, and a few recent papers) must be confirmed against the
live record before manuscript submission.
