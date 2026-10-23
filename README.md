# IdealAorta-PINN

Code for the sparse-velocity reconstruction study in

> *Effects of Inlet Diameter and Sac Asymmetry on Transitional Flow in Idealized Descending
> Thoracic Aortic Aneurysms, with Neural-Field Recovery of Pressure and Wall Shear Stress from
> Sparse Velocity* (manuscript under review).

Velocity is observed at about 1.3% of the interior nodes of a transient CFD solution at two
instants 5–10 ms apart. From these observations each method recovers the velocity at the other
nodes, the gauge pressure and the wall shear stress (WSS) on the aneurysm zone. Space-time
neural fields (with and without a continuity penalty, and physics-informed networks with the
momentum balance in the loss) are compared with linear and radial-basis-function (RBF)
interpolation. Every method passes through the same momentum integration and near-wall shear
estimate, so the comparison isolates the reconstruction itself.

## Contents

- [Data](#data)
- [Setup](#setup)
- [Manuscript and Overleaf](#manuscript-and-overleaf)
- [Reproducing the results](#reproducing-the-results)
- [Outputs](#outputs)
- [Code layout](#code-layout)

## Data

Twelve idealized descending thoracic aortic geometries were simulated with transient, rigid-wall
CFD (ANSYS Fluent, transition SST k–ω model, Newtonian blood with ρ = 1050 kg m⁻³ and
μ = 3.5 × 10⁻³ Pa s):

| Group | Inlet diameter (cm) | Sac shape | Cases |
| --- | --- | --- | --- |
| Aneurysmal | 2.0 / 2.3 / 2.6 | axisymmetric, anterior-dominant or posterior-dominant; 4.0 cm maximum diameter | 1–9 |
| Control | 2.0 / 2.3 / 2.6 | smooth taper to 80% of the inlet diameter | 10–12 |

The CFD exports are not distributed with this repository; they are available from the
corresponding author on reasonable request. The code expects them at

```text
data/raw/full_2026-09/Case <n>/<time>.csv
```

Each file is an ANSYS CFD-Post export of one case at one instant with five blocks: the fluid
volume (velocity, pressure, eddy viscosity, k, ω at every node), the lumen wall and its
aneurysm zone (with the wall-shear vector), the inlet face and the outlet face. The instants
are 1.775, 1.780, 2.390 and 2.400 s for every case (Case 2 has 1.788 s instead of 1.775 s),
plus each case's original systolic export time.

The CFD figures also use three smaller exports, available on the same terms: the solver's
time-averaged wall shear (`data/raw/TAWSS 30.09.2026/`), the maximum principal stress of the
structural model (`data/raw/MPS/`) and the plane-averaged speed and k on planes D1–D8
(`data/results_on_slices.csv`).

## Setup

Python ≥ 3.10 and PyTorch (a CUDA build for training; the interpolation comparators and the
analyses run on CPU):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your CUDA version
pip install -r requirements.txt
```

All commands are run from the repository root as `python main.py <command>`;
`python main.py -h` lists them and `python main.py <command> -h` gives their options.

## Manuscript and Overleaf

The current manuscript is `doc/IdealAorta_EACFM/main.tex`, updated from the Overleaf
export supplied on 22 October 2026. The supplied bibliographies are in `bib/`;
supporting tables and figures are in `tables/` and `figures/` in the same directory.
The downloaded duplicate source, duplicate bibliographies and LaTeX build products
have been removed from the manuscript directory.

The revision addresses Abaid's four questions: the three data-only endpoint
advantages, a qualified acceleration-sensitivity explanation checked against the
saved Brosnan records, the all-nine reference audit and its overlap with the
confirmation geometries, and the maximum of the displayed Figure 13 stress field.
CFD-related edits use `\cfdrevision{...}` and appear in red for coauthor review.

Figure 4 distinguishes direct development momentum-PINN pressure from the
confirmatory velocity-to-pressure integration route. Figure 5 omits the redundant
numbered stage strip. The study protocol is explained in Section 3.8 and the
Discussion. The signed axial WSS profiles in Figure 11 use a reduced width,
with body text following it on the same page and forced gaps removed around the neighbouring figures.
Scientific wording distinguishes
regional stress medians from local maxima, TAWSS medians from upper percentiles,
and dependent draw counts from independent statistical evidence.

The full manuscript and selected primary papers were reviewed on 22 October.
`report/review_2026-10-22.pdf` contains the Saitta explanation, reviewer comments,
evidence priorities, Abaid's 22 October responses and remaining checks; its editable source is
`doc/review_2026-10-22.tex`. The manuscript now discusses the prior SIREN work
more explicitly and cites acquisition-aware synthetic MRI and structural
material/thickness sensitivity studies. Author-reported solver settings and their
remaining gaps are recorded in `doc/author_clarifications_2026-10-22.json`.
The red methods edits distinguish pressure-only Static Structural loading at
1.775 s from common flow exports at 1.780 s, specify first-order implicit time
integration and sac/neck inflation, and distinguish supplied plane area averages
from node-weighted whole-domain calculations. No dedicated solid-mesh convergence
study was performed; native projects and transfer/export verification remain
unavailable for independent inspection.

`experiments/measurement_realism_2026_10_22/` now contains an isolated pilot runner,
separate observation values, differentiable spatial averaging, operator-aware
scalar and globally divergence-free vector RBFs, and matched-count removal controls.
Every method retains the same observation hold-out. Volume boxes are certified
against a declared point-cloud geometry proxy, not the unavailable native CFD
surface. Local numerical checks pass; dependency-backed verification and CFD
execution are recorded separately on Brosnan. See its `README.txt` and `pilot.json`
for smoke, feasibility and full development stages. The independent evaluation
design is specified in `independent_protocol.json` and must be revised and sealed
before evaluation access. The experiment goal is paused following the 23 October
audit, and automatic feasibility/final pilot launches have been disabled. The
existing capacity sweep continues unchanged; no full pilot or independent
evaluation has completed. Historical experimental configurations and numerical
results are retained. Manuscript wording, including the captions in
`tables/reference_floor.tex` and `tables/robustness.tex`, has changed.

`doc/revision_audit_2026-10-23/` preserves Abaid's reconstructed 22 October
Overleaf source, the supplied audit, the original comparison PDF and source,
and the controller-stop receipt. The original comparison uses blue additions
and red deletions; `main_red_review.tex` builds a coauthor copy with additions
in red and deleted wording hidden. It is a review document, not a replacement
for the canonical `main.tex`. Pending scientific and wording decisions are
listed in `audit_response.txt` in that directory.

`figures/generated/confirm_p_sys.dat` and `confirm_p_dia.dat` contain the pressure
scatter-plot coordinates read directly by PGFPlots during compilation. They are
required figure inputs; keep them alongside the PDF and PNG figures.

Keep `main.tex` as the main document and use pdfLaTeX. The current coauthor
review is `report/main_red_review.pdf` (54 pages), with revisions in red and
deleted wording hidden. Its source and preserved comparison are in
`doc/revision_audit_2026-10-23/`. The older `report/main.pdf` predates the
23 October Hager bibliography correction. The two previously missing
bibliography entries are included in the sources.
The obsolete duplicate data-availability statement was removed; the latest
coauthor-supplied policy text is retained. Compare subsequent coauthor changes
before replacing the live Overleaf root file.

Previous drafts, audit evidence and the removed/replaced pre-cleanup files are retained
in `report/research_archive.zip`; `cleanup_manifest.json` inside it records paths
and SHA-256 hashes. Raw data remain under `data/`; the active replacement Case 8
export is `data/raw/MPS/Case 8.xls`. Keep temporary builds and verification renders
outside the manuscript source directory and remove them after checking.

To compile locally from the repository root in PowerShell:

```powershell
latexmk -cd -pdf "-outdir=$env:TEMP/IdealAorta-latex" doc/IdealAorta_EACFM/main.tex
```

## Reproducing the results

Each study is a YAML file in `configs/reconstruction/`. `main.py jobs <study> --kind <kind>`
writes one command line per run, and `main.py run-jobs <file>` runs such a file with bounded
concurrency (GPU jobs are spread round-robin over `--gpus`; pass `--gpus` with no value for
CPU-only work). The steps below follow the sections of the paper.

**1. Prepare the data (once).** Splits the CSV exports into per-block parquet files
under `data/processed/full/`.

```bash
python main.py prepare
```

**2. Checks of the complete CFD fields (Section 5.1, Table 10).** Momentum-budget and pressure
checks of each case and phase, and the WSS estimators applied to the complete CFD velocity.

```bash
python main.py jobs reference --kind reference > jobs_reference.txt
python main.py run-jobs jobs_reference.txt --parallel 4 --gpus
```

**3. Development test on Cases 1 and 9 (Section 5.2).** First the physics weight of each
neural arm (10 000 steps, Case 1 at systole), then the prespecified comparison of the four
neural arms with linear and RBF interpolation.

```bash
python main.py jobs dev_select --kind train > jobs_select.txt
python main.py run-jobs jobs_select.txt --parallel 4 --gpus 0
python main.py select dev_select                         # -> w = 0.01 for every physics arm

python main.py jobs dev_main --kind train > jobs_dev_train.txt
python main.py jobs dev_main --kind baseline > jobs_dev_baseline.txt
python main.py run-jobs jobs_dev_train.txt --parallel 4 --gpus 0
python main.py run-jobs jobs_dev_baseline.txt --parallel 4 --gpus
python main.py summarize dev_main
python main.py diagnose <run name> [<run name> ...]     # momentum PINNs through the shared post-processing
```

**4. Confirmation on Cases 2–8 (Sections 5.3–5.5).** The continuity field and the data-only
field, linear and RBF interpolation and the post-processing floor, at three draws of the
2.5 mm grid and one draw each of the 1.5 and 4.0 mm grids.

```bash
python main.py jobs confirm --kind train > jobs_train.txt
python main.py jobs confirm --kind baseline > jobs_baseline.txt
python main.py run-jobs jobs_train.txt --parallel 4 --gpus 0
python main.py run-jobs jobs_baseline.txt --parallel 4 --gpus
python main.py confirm confirm          # prespecified hypothesis tests (expect 14/14 units, medians 0.46 and 0.51)
python main.py report confirm           # CSV tables and the result figures
```

**5. Robustness to the interpolation comparator (Section 5.6).** Tuned and space-time RBF,
draw 0 of the 2.5 mm grid; `confirm` then includes them in its per-unit table.

```bash
python main.py jobs confirm --kind robustness > jobs_robustness.txt
python main.py run-jobs jobs_robustness.txt --parallel 4 --gpus
```

**6. Sensitivity to near-wall observations (Section 5.7).** The continuity field and RBF,
draw 0 of the 2.5 mm grid, without observations within 1.25 mm of the wall. These runs carry
the suffix `_mw1.25` and are kept out of the confirmatory analysis.

```bash
for c in 2 3 4 5 6 7 8; do
  w="--case $c --target 1780 --grid 2.5 --seed 0 --min-wall-mm 1.25"
  [ $c = 2 ] && t="1780 1788" || t="1775 1780"
  python main.py train --arm cont --wphys 0.01 --times $t $w
  python main.py baseline --method rbf --times $t $w
  python main.py train --arm cont --wphys 0.01 --times 2390 2400 ${w/1780/2400}
  python main.py baseline --method rbf --times 2390 2400 ${w/1780/2400}
done
```

**7. Figures.** `report` (step 4) draws the result figures from the run records. The CFD-field
figures and the reconstruction maps need the CFD exports, and the maps also the trained fields:

```bash
python main.py figures cfd maps
```

Every figure is a 300 dpi PNG in `report/figures/`. All plotting settings (fonts, sizes, colors,
markers, color scales) are in the configuration block at the top of `idealaorta_pinn/plots.py`.

**Compute.** A 20 000-step neural-field fit took a median of 23 min (continuity field) and
17 min (data-only field) on one NVIDIA Quadro RTX 8000 shared with other jobs, using about
2.5 GB of GPU memory. The confirmation has 140 such fits. The interpolation comparators run on
CPU in a few minutes each (the tuned and space-time RBF take longer because they search their
settings on the withheld observations).

## Outputs

```text
models/<run>/              trained neural field (model.pt) and its training history
report/runs/<run>.json     one record per reconstruction: method, problem, settings and scores
report/masks/              observed node indices of each observation draw
report/checks/             audit_*.json and wss_*.json (step 2), diag_*.json (diagnose)
report/tables/             study summaries (JSON) and report CSVs:
                             confirm_units.csv    mean error per case-phase unit and method
                             confirm_medians.csv  median over cases per phase, grid and method
                             confirm_tests.csv    the prespecified hypothesis tests
report/figures/            300 dpi PNGs:
                             confirm_ratios, confirm_spacing, confirm_pressure_forms   (report)
                             bc_profiles, streamlines_2p0/2p3/2p6, wss_systole, wss_diastole,
                             tawss, mps, xwss_lines, plane_profiles                    (figures cfd)
                             hidden_velocity, field_maps_c04_t1780, field_maps_c07_t2400 (figures maps)
report/logs/               one log per queued job and done.txt (exit code per job)
```

Run names encode the problem, for example `cont_oracle_c04_t1780_g2.5_s0_i0_w0.01` is the
continuity arm with the CFD eddy viscosity in its residual, Case 4, target 1.780 s, 2.5 mm
grid, observation draw 0, network seed 0 and physics weight 0.01; `rbf_c04_t1780_g2.5_s0` is
the RBF comparator on the same problem. Errors are relative L2 errors: velocity on the
unobserved interior nodes, pressure on the interior nodes more than 1 mm from the wall (after
a common gauge), WSS magnitude on the aneurysm-zone wall nodes.

## Code layout

```text
main.py                      command-line entry point (all commands above)
idealaorta_pinn/
  config.py                  paths and fluid constants
  data.py                    CFD-Post export parsing, parquet cache, snapshot loaders, wall normals
  problem.py                 one reconstruction problem: case, window, observation grid, hidden targets
  fields.py                  space-time neural field and its momentum and continuity residuals
  training.py                fit and score one neural-field arm; diagnostics of trained arms
  interpolation.py           linear, RBF, tuned RBF and space-time RBF comparators
  postprocess.py             least-squares gradients; velocity -> pressure and wall shear; scores
  reference.py               checks of the complete CFD fields (momentum budget, pressure, WSS)
  study.py                   study configs -> jobs; weight selection, summaries, hypothesis tests, report
  plots.py                   every figure, with all plotting settings (fonts, colors, scales) at the top
configs/
  constants.yaml             fluid properties
  reconstruction/            reference, dev_select, dev_main and confirm study definitions
```
