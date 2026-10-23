Measurement-realism development study, 22 October 2026

Status and evidence
-------------------
The isolated study now has a CFD adapter and runner. It includes separate
immutable observations, differentiable volume averaging, operator-aware scalar
and global divergence-free vector RBFs, and matched-count removal controls.
The experiment goal is PAUSED following the supplied 23 October audit. The
waiting pilot controller was stopped before any stage started. The existing
capacity sweep remains running unchanged; no new fits or benchmarks are being
launched. The full scientific pilot and independent evaluation are NOT complete. The first
CFD smoke run completed eight fits with clean targets unchanged. One full-scored
2,000-step run was retained from an early sweep, then that sweep was deliberately
stopped when its reduced Gaussian comparator failed capacity/boundary checks.
Both cases at both phases have now been prepared with all native wall nodes. Execution
state is established by process handles, logs and terminal receipts on Brosnan,
not by this README. The current code_v22 snapshot passed all 66 checks on Brosnan,
including both CPU factor backends, CUDA products/factors, neural training paths,
pressure-solver equivalence and cap quadrature. The v22 development adapter
completed eight fits, and its actual assessment passed the external-contract
validator. The unchanged v21 capacity sweep is running. The current 128-record
feasibility and 384-record final development stages have zero records. Their
automatic controller is disabled. The expanded comparison requested in the
audit needs a revised design and record count before execution. Every check
skipped locally passed on Brosnan; all 66 passed there with zero skips.
The test receipts are saved under report/measurement_realism_2026_10_22/checks/.

Brosnan execution root:
  /home/olarinoyem/measurement_realism_2026_10_22/
Historical inputs, imported read-only:
  /home/olarinoyem/aorta_rescore/
Scientific Python:
  /home/olarinoyem/miniconda3/envs/deep_tf/bin/python
The adapter supports the flat local package and nested Brosnan reconstruction
package. No historical source, configuration, mask, model or score is overwritten.
Code snapshots and all failed tests/attempts are retained in the dated root.

What the experiment measures
----------------------------
* Point velocities and physical full-box averages use the same noise realization
  at matched core locations. The current grid is snapped to CFD nodes, as in the
  original study. It is not an exact MRI voxel-centre grid.
* Averages are physical-volume Gauss-Legendre integrals of a frozen piecewise-
  linear CFD interpolant, not averages over an irregular collection of mesh nodes.
  The same quadrature is applied to neural predictions and RBF design matrices.
* Original surface connectivity/CAD is unavailable. Oriented boundary points and
  inferred inward normals define a geometry proxy. Every full box is certified
  within THAT proxy by checking all boundary points that can be nearest anywhere
  in the box. This does not prove equivalence to the native CFD lumen. A 0.1 mm
  geometric margin, coverage counts and this limitation are saved in the receipt.
* Orders 3 and 5 are compared. If their relative velocity-average difference is
  at most 1% at both times, order 3 is used. Otherwise order 5 must pass against
  order 7. Failure stops volume preparation; it is never silently accepted.
* Gaussian velocity noise is specified in m/s with independent spatial/component
  draws, a separate seed and explicit cross-time correlation. This is a kinematic
  sensitivity, not complex-signal MRI simulation, partial volume or aliasing.

Fair fitting and scoring
------------------------
Exactly 10% of base observation locations are held out at both times. Every
method uses the same split and retains its hold-out without an all-observation
refit. This corrects the fit-information mismatch for the NEW study; historical
neural fits retained their hold-out while final interpolants used all samples.

Near-wall removal uses the 1.25 mm criterion. Its random-removal control retains
exactly the same numbers in BOTH fitting and validation strata. Removed arms
inherit the original noise realization. The original hidden scoring node set is
fixed across removal and averaging conditions, so changing available locations
does not change the target against which methods are compared.

The reduced Gaussian scalar/vector RBFs share length-scale and ridge grids and
geometry-only farthest-point centres. Their adequacy checks failed at up to
1,024 centres; those trials are retained and are not final comparator rankings.
The current candidate uses sparse C4-Wendland kernels plus a low-order
affine background (trace-free for the vector field). All available native wall
nodes are the default constraint set for EVERY method. A 2,048-wall-node pool
was an early computational feasibility shortcut and is now a labelled sensitivity
only. The sparse comparator is still being checked, not declared adequate.
The retained Gaussian branch uses bounded dense designs and LSQR; the compact
branch supports LSMR and an audited direct ridge solver. The current native
capacity sweep selects the direct solver explicitly. Numerical diagnostics,
failed candidates and timing are retained. The vector kernel and trace-free
affine extension are globally solenoidal. There is no local
nearest-neighbour switching or ordinary partition blending. Reduced-centre
capacity and convergence still need development-case checks before ranking it
as a suitable full-study comparator. This is a new comparator, not a rerun of
the historical local thin-plate RBF.

The neural architecture is the historical Fourier-feature FieldNet. Its loss
uses averages for averaged observations, known no-slip/inlet information, and
optionally the continuity penalty. Initialization and sampling streams are
explicit. Terminal step counts are fixed; no hidden target selects a checkpoint.

The full scorer reuses the historical 40-neighbour gradient, pressure sub-cloud
>1 mm from the wall, outlet gauge, WSS offset 0.25 mm and four pressure forms.
Pressure with CFD eddy viscosity is labelled ORACLE. Clean targets are read-only
and their hashes are checked after every fit. Flux is not inferred from node
averages: original face-area weights are absent. Planar-cap convex-hull triangulation
now supplies verified area quadrature, labelled as a reconstructed-cap
approximation. It reports inlet/outlet imbalance, not full closed-wall flux.

Stages (do not conflate them)
----------------------------
1. Integration smoke: Case 1 systole, 100 steps, selected point/average conditions,
   velocity diagnostics only. No pressure/WSS conclusion or method ranking.
2. Bounded feasibility: Cases 1 and 9, both phases, one draw, 2,000 neural steps,
   eight conditions and four methods = 128 fully scored records. Establish cost,
   numerical convergence and whether the proposed comparator is adequate.
3. Full exploratory development pilot: all three paired draws, 20,000 neural
   steps, eight conditions and four methods = 384 fully scored records. The
   final audit requires numerical, boundary and scoring adequacy for every arm. More
   provisional noise/correlation levels are secondary extensions, recorded before
   they are run. All results remain post hoc development evidence.

Verification and commands
-------------------------
From this directory in a scientific environment, with the historical root on
PYTHONPATH:
  python -m unittest discover -p "test_*.py"

Runner (an OUTSIDE output directory is mandatory):
  python run_pilot.py --historical-root /home/olarinoyem/aorta_rescore \
    --output /home/olarinoyem/measurement_realism_2026_10_22/feasibility_NEW \
    --cases 1 9 --targets 1780 2400 --steps 2000 --score full

The default comparator family is the full-centre sparse candidate. The --centres
argument applies only to the retained reduced Gaussian branch. Select final
support/ridge/iteration choices from recorded observation-only and known-boundary
checks; a successful numerical test does not establish CFD adequacy. Never change code or
settings inside a completed/running output root: its protocol/source hashes must
match exactly. Use a fresh code snapshot and output root when they change.

For the full pilot use separate draw00/draw01/draw02 subdirectories. Grid/init
seeds are 0/1/2; noise seeds 22001/22002/22003; declare split seeds separately.
The audit refuses to call one feasibility draw a complete three-draw pilot:
  python assess_pilot.py /path/to/full_pilot
  python assess_pilot.py /path/to/one_draw_feasibility --stage feasibility

Independent evaluation
----------------------
independent_protocol.json specifies two tracks: external velocity reconstruction
using the public 4DFlowNet release, and independent pressure/WSS evaluation on new
native CFD exports. The design must be sealed with the finished source/config,
capacity checks and a prior-access declaration before evaluation targets are used.
The runner is restricted to Cases 1 and 9 and cannot accidentally use new cases.

All twelve existing geometries have complete-field audit records, including
controls. Another seed, noise realization or time cannot restore geometry-level
independence. The public release has no independent pressure/WSS truth. Neither
manufactured tests nor a derived pressure reference supplies that missing evidence.

Kernel provenance
-----------------
McNally CP (2011), Divergence-free interpolation of vector fields from point
values - exact div B=0 in numerical simulations. MNRAS Letters 413:L76-L80.
https://doi.org/10.1111/j.1745-3933.2011.01037.x
https://arxiv.org/abs/1102.4852
The full paper was read. The 3-D kernel is derived from its matrix principle;
the author's correction to equation 7 is noted at https://www.colinmcnally.ca/dfi/.
This is numerical MHD precedent, not blood-flow validation. A Saitta-matched SIREN
and acquisition pipeline remain a separately declared future comparison.

Additional diagnostics and sealing
----------------------------------
Temporal error decomposition reports snapshot bias, cross-time covariance and
acceleration error, with its exact second-moment identity checked numerically.
It is a post hoc diagnostic, not proof of a physiological mechanism. Bulk field
predictions are reused during scoring; an equivalence test checks this cache.

The audited pressure integrator uses the same historical system, gauge, SciPy
LSQR tolerance and iteration limit. Stop codes, residuals, condition estimates
and time are now retained. Missing or failed convergence cannot pass the final
audit, even if a score was written.

seal_protocol.py refuses to seal a smoke run or incomplete development pilot.
It requires comparator adequacy receipts for all four development units, an
explicit prior-access declaration, matching core source hashes, and all declared
development runs. Sealing a design is not performing its independent evaluation.

Sparse-kernel source: Lowitzsch (2002), Chapter III, equation 3.5 and Lemma 3.5:
https://oaktrust.library.tamu.edu/bitstreams/7f87c44d-9bf2-47e0-81ec-7bb7fc3b8475/download
Only the relevant construction was consulted; no full-thesis reading is claimed.
LSMR implementation reference (installed SciPy version):
https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.sparse.linalg.lsmr.html

Comparator diagnostics added after the initial full-wall check
-------------------------------------------------------------
The centre-separation option thins basis centres by geometry alone. It keeps
EVERY native wall fitting row and is distinct from wall-constraint subsampling.
The diagnostic support grid extends to 10/15/20 mm; compact support radius and
Gaussian length scale are different kernel parameters and must not be equated.
Known wall and prescribed inlet residuals can gate candidate selection before
ranking by the shared observed-velocity hold-out. Failed candidates retain
structured trial records, including numerical nonconvergence.

Optional column scaling solves [A D; sqrt(lambda) D] z = [b; 0], x=D z.
It preserves the original ||A x-b||^2+lambda||x||^2 objective. A numerical
equivalence test compares both solvers with the exact ridge solution; the
original-coordinate stationarity residual is retained. Speedup and CFD adequacy
must be established by actual runs; they are not inferred from that test.

The final audit recomputes hashes/counts from saved observation arrays, checks
known wall AND inlet residuals at both times for every method, and checks that
capacity receipts cover both comparator families and both challenging operators.
A partial successful comparator check cannot seal the external protocol.

External adapter development
----------------------------
The public release is 10.17608/k6.auckland.24424888.v1 (CC BY 4.0). Only
aorta01_HR.h5 is used for adapter development. Its 71 timeframe labels run from
1 through 71; aorta_model=1 and boundary_cond=0 are identifiers, not numeric
boundary data. The mask is fractional: the primary threshold is 0.6, with
geometry-only coverage checks at 0.5 and 0.9. At primary threshold the three
draws yield 5063/5090/5064 point centres and 3040/3049/2982 complete boxes in
the conservative interpolation-cell mask proxy. These counts depend on the
publication-derived 0.594 mm spacing assumption. The file has no stored
spacing, affine or component-axis metadata. The release checksum and the
eight-frame development axis audit passed. Anatomical orientation is not
independently established.

The selected external contract is measurements-only velocity robustness, with
no invented no-slip, inlet or outlet pressure targets for any method. Both
neural and RBF implementations have explicit measurements-only modes. Native
Cases 1/9 retain their known wall and inlet information. The public release
supplies no independent pressure/WSS truth. Evaluation velocity arrays have
not been downloaded or inspected. Exact budgets/settings are now specified in
independent_protocol.json, but its source/design seal remains incomplete.

The current eight-fit adapter check passed. Its assessment includes the direct
solver hash and passes the actual external-contract validator. A manufactured
regression also rejects a changed solver hash. These checks establish adapter
integration and provenance, not native comparator capacity or accuracy.

Primary sources for adapter provenance:
https://doi.org/10.17608/k6.auckland.24424888.v1
https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2020.00138/full
https://github.com/EdwardFerdian/4DFlowNet/tree/c1d452e312f5a99034fe52da34805f6af4434d08

Direct solver and basis capacity
--------------------------------
DirectRidgeSystem forms A.T A in column blocks without materializing dense A.
It preserves ||A x-b||^2+lambda||x||^2, audits stationarity against the original
sparse A/B, and supports iterative refinement. Explicit CPU/GPU allocation
bounds precede large allocations; dense work arrays remain in RAM.

The direct path uses row-contiguous CUDA blocks and can stream Gram products
to the host for CPU factorization. CPU factorization explicitly selects SciPy
or Torch, with the choice retained in execution contracts and fit receipts.
The native vector case reproduced a segmentation fault inside SciPy
cho_factor. Torch MKL completed that candidate and passed its original-system
stationarity and known-boundary checks. This identifies a working alternative;
no specific underlying library defect is asserted.

compact_check.py, compact_capacity.py and run_public_pilot.py accept
--solver-method direct; run_pilot.py uses --rbf-solver-method direct. All expose
--direct-factor-device, --direct-cpu-factor-backend, --direct-block-columns,
--direct-max-gpu-gib and --direct-max-host-gib. Existing runners retain LSMR as
the default unless the direct path is explicitly selected.

compact_capacity.py retains every observation-fit, native wall and inlet row.
Only basis centres are thinned by geometry. Both finest increasing capacities
must pass numerical and known-boundary gates. Their absolute relative change
in observed validation error AND paired held-out predictions must each be
<=2%. This concerns observed-data capacity, not hidden pressure/WSS accuracy.
Each native case/phase requires both scalar/DF families and noisy point/volume
operators. Sealing verifies model/prediction hashes and recomputes the plateau.
A single successful candidate or boundary-only receipt cannot certify capacity.

The current sweep compares 1.0/0.8 mm basis separations, 15/20 mm support radii
and ridges 1e-7/1e-8/1e-9. Its bounds are 160 GiB host, 42 GiB GPU and 1.2
billion nonzeros. The earlier 1.25/1.0 mm scalar basis check remained unsettled;
its results and all superseded or failed attempts are preserved. Seven critical
comparator/observation/capacity files are unchanged between v21 and v22, so the
live v21 fits remain valid for the v22 reporting and handover implementation.

Execution, evidence and resumption
----------------------------------
The local isolated suite passes 58 of 66 tests; eight require dependencies in
the native environment. All 66 pass on Brosnan with zero skips. The v22 adapter
completed eight fits and the corrected actual handover passed. Current counts
and worker handles are recorded in the ignored study report directory:
  report/measurement_realism_2026_10_22/progress.json
  report/measurement_realism_2026_10_22/checks/remote_v22_20261023/

The prepared-array audit verified all eight conditions for each of the four
native units: saved hashes/counts, inherited shared hold-outs, paired noise,
full known-wall constraints and quadrature checks. Near-wall and random removal
retain exactly matching fitting/validation counts: 3637/418 for Case 1 and
5306/583 for Case 9. These are preparation checks, not completed method fits.

Both native_pipeline_v22.json and native_pipeline_v22_threads8.json controllers
are stopped. Their plans are retained as history and must not be resumed
automatically. The exact old commands and seeds are retained in the study
report directory. Grid/init seeds
are 0/1/2, split seeds 42000/42001/42002, noise seeds 22001/22002/22003 and
removal seeds 32001/32002/32003. Native boundary seed 52001 is fixed.

The short 2000-step stage retains any neural boundary inadequacy and can then
proceed to the already declared 20000-step budget. Missing records, comparator
failures, information mismatches or unconverged pressure solves stop that
handover. The final 384-record audit retains the strict 2% boundary gate for
every method. No failed final audit can produce a sealed independent protocol.

After a complete final audit, the controller writes every case/phase/draw score,
within-geometry means and paired condition differences. WSS reporting includes
absolute magnitude/vector RMS Pa on the historical aneurysm-zone mask and with
the same estimator. The report keeps dependent phases/draws within geometry;
it makes no population significance or independent-geometry evaluation claim.
The controller may seal the design only after all declared gates pass. It never
downloads evaluation files or executes external evaluation.

Current limits are eight hours per capacity unit, 36 hours of dependency wait
and 48 hours per pilot stage. GPU 1 is used; dense arrays never spill to the NAS.
Completed/running output roots and source snapshots are immutable. An observation
timeout is not a terminal result: inspect the same PID and terminal record before
any restart. Historical sources, test failures, probes and stopping reasons are
retained under the dated execution root. The old v11 worker is confirmed terminal;
normal university SSH to Brosnan is restored. There is no remaining SSH blocker.
