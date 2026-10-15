# Research manuscript review

Reviewed 15 October 2026. Source: `main.tex` in `C:/Users/olarinoyem/Downloads/IdealAorta_EACFM`; accompanying tables and selected reconstruction implementation/configuration in `C:/Users/olarinoyem/Desktop/sms/IdealAorta-PINN`.

## Assessment

**Major revision before submission.** The reconstruction study has a clear and useful question: which parts of pressure and wall-shear recovery depend on the velocity representation, physics penalties, temporal information and closure inputs? Reporting a negative momentum-PINN result, a data-only neural comparator, common post-processing, stronger RBF checks and near-wall sensitivity makes this more informative than an unqualified PINN success story.

The principal weaknesses are inconsistent scientific interpretation and incomplete provenance for parts of the CFD/structural analysis. These are more consequential than whether the prose sounds AI-assisted. The revised copy improves the language and corrects statements supported by the supplied tables and code. It is an editorial working draft, not evidence that the outstanding modeling questions have been resolved.

The raw CFD exports, trained models, per-run result records and ANSYS projects are absent from these two workspaces. I did not rerun the simulations or reconstruction experiments. The numerical checks below establish consistency with supplied tables, not independent replication of the reported results. The review includes selected reference spot-checks, not verification of every cited paper or every comparison in the literature table.

## The screenshot

The original is clearer than the proposed paraphrase. In particular, "radius loaded" is not an improvement on "loaded radius," and "the median segment of the controls" obscures that the quantity is a stress median over a segment. Name the quantity explicitly and separate the result from its mechanical explanation:

> Median sac-body MPS on the wall opposite the dominant bulge increases with inlet diameter. The controls show the same trend in segment-median MPS (52.5, 58.4 and 65.1 kPa for inlet diameters of 2.0, 2.3 and 2.6 cm, respectively).

If an explanation is included, qualify it:

> This trend is consistent with greater tensile loading at a larger vessel radius under comparable pressure and wall thickness.

The three control numbers agree with `tables/mps_summary.tex`. They are not a justification for claiming that a particular wall site will rupture.

## Priority findings

### 1. Fluid/structural coupling is described inconsistently

Original locations: lines 727-736, Figure workflow at 808-818, line 828, and lines 1288/1483.

The methods claim bidirectional FSI. The workflow depicts wall-pressure transfer from Fluent to Mechanical, while the reconstruction section explicitly states that all flow results come from rigid-wall CFD. Later limitations describe a quasi-static structural calculation whose coupling-step and mesh-update settings were not retained. A pressure-transfer diagram alone cannot establish a two-way moving-wall simulation.

**Revision made:** describe the retained evidence as rigid-wall flow plus pressure-loaded structural stress, explicitly noting that bidirectional coupling is not established by the available records.

**Author action:** recover the ANSYS project or logs. Specify whether loading was one-way, two-way, static or transient; pressure-transfer time; external pressure; reference configuration/prestress; coupling iterations; deformation feedback; and solid-mesh convergence. If those records are unavailable, keep the narrower description and consider moving the structural results to supplementary material. Do not invent missing settings.

### 2. The TAWSS trend contradicts its table

Original line 1266 and `tables/tawss_summary.tex`.

The prose says increasing inlet diameter progressively reduces TAWSS. That is true of the control segment medians and aneurysmal upper percentiles, but not the aneurysmal segment medians:

| Morphology | 2.0 cm | 2.3 cm | 2.6 cm |
| --- | ---: | ---: | ---: |
| Axisymmetric median, Pa | 0.51 | 0.70 | 0.70 |
| Anterior-dominant median, Pa | 0.52 | 0.67 | 0.73 |
| Posterior-dominant median, Pa | 0.53 | 0.83 | 0.69 |
| Control median, Pa | 4.53 | 3.30 | 2.51 |

**Revision made:** distinguish segment medians, upper percentiles and controls. The posterior-dominant medians are not monotonic. Also remove the inference that a low cycle average plus one low systolic value proves continuously low shear throughout the cycle.

### 3. Case 9 has the highest reported sac-body median, not an established global peak

Original lines 1286-1288 and `tables/mps_summary.tex`.

Case 9 has the highest listed anterior sac-body median, 80.3 kPa. Case 5 has the highest listed neck 95th percentile, 113.4 kPa, compared with Case 9's 112.9 kPa. The table does not report actual point maxima. Neither number establishes a rupture site.

**Revision made:** name the specific statistic and remove the claim that Case 9 has the highest MPS of all sites and cases. Keep the distinction between modeled loading and failure. Local strength, wall thickness variation and a validated structural model would be needed for failure assessment.

### 4. The draw-level sign-test p-values assume independence the study does not have

Original line 1343.

The three observation/initialization draws share each geometry, and the two phases also share anatomy. Treating 42 comparisons as independent replicates produces an unjustified inferential claim. The manuscript already says no significance test was planned and acknowledges dependence; adding a very small p-value still invites misinterpretation.

**Revision made:** retain the descriptive counts (14/14 case-phase means; 42/42 WSS draws; 39/42 pressure draws) and remove the p-value. Seven designed geometries are also not a random patient sample. Any additional inference should define a defensible independent unit and uncertainty target before analysis.

### 5. Evaluation geometries informed the shared post-processing

Original lines 1196-1198.

The complete pressure and WSS fields of Cases 2-8 were used to choose the momentum form and WSS target region before the reconstruction comparison. This is disclosed and differs from tuning a reconstruction on its hidden labels, but it means the final cases were not untouched by the entire pipeline-design process. Applying a common postprocessor to every method does not prove that its design is neutral to every representation.

**Revision made:** state that the evaluation is held out from reconstruction tuning rather than from all analysis choices.

**Author action:** preserve the written protocol and dated decisions. The current YAML file labels its freeze date and hypotheses, but comments alone do not independently prove the historical sequence. Present the comparison as an internal benchmark with reference-informed post-processing; an entirely untouched set would support a stronger validation claim.

### 6. The final fitting subsets differ across methods

Original observation-design paragraph at line 864; code in `training.py` and `interpolation.py`.

The neural fields fit approximately 90% of the sampled observations. Fixed linear/RBF comparators fit all sampled observations. Tuned RBF variants use a holdout for selection and then refit on all observations. Thus they share the sampled observation pool and boundary values, but not exactly the final fitting subset. The present code gives interpolation more fitting observations, so this is not evidence of an unfair neural advantage; it is a reporting inconsistency.

**Revision made:** describe the actual allocation. Also correct the neural speed-scale definition to use training observations, and describe the implemented RBF singular-system fallback as 64 -> 128 -> 256 sources.

### 7. The evidence favors the neural representation more strongly than the continuity penalty

Original Table reconstruction summary and lines 1420, 1467-1472.

At the primary spacing, data-only systolic WSS error is 0.333 versus continuity's 0.356, and data-only diastolic pressure error is 0.063 versus 0.094. Their diastolic WSS errors are also nearly equal (0.161 and 0.163). The continuity penalty is most consistently useful at the sparsest sampling. The existing discussion largely acknowledges this and should remain central to the abstract, title framing and conclusion.

**Revision made:** preserve the negative momentum-PINN result and explicitly attribute most of the gain to the neural representation. Limit the robustness claim to the interpolation schemes actually tested; no divergence-free non-neural comparator was evaluated.

### 8. Imaging and clinical interpretation need strict limits

Original lines 436-442, 864, 1268, 1288, 1453-1455 and 1487.

Noise-free mesh-node point samples are not 2.5 mm MRI voxel averages. Some observed points are 0.03 mm from the wall. Removing observations within 1.25 mm increases continuity-field WSS error from 0.356 to 0.806 at systole and from 0.173 to 0.521 at diastole for the single-draw sensitivity experiment. These absolute errors should accompany any statement about practical WSS recovery. Pressure results using CFD eddy viscosity are conditional on unavailable-in-imaging closure information.

The introduction also mixes abdominal, cerebral and descending-thoracic findings, claims improved rupture prediction without validating a risk model, and treats a 5 cm threshold as general. The cited 2022 ACC/AHA guideline distinguishes aortic segments and clinical risk modifiers; a generic threshold is inappropriate here. This is a literature-framing issue, not patient-specific guidance.

**Revision made:** remove the universal threshold, outdated prevalence paragraph and unvalidated risk claims; identify anatomical evidence more precisely. Keep the oracle and near-wall limitations visible.

### 9. Numerical verification is narrower than the claims may imply

Original lines 698-723, 828, 734 and 1483.

The reported WSS mesh sequence is 47.8 -> 55.4 -> 67.8 -> 67.2 Pa. It is non-monotonic, and the GCI assumes p=2 rather than demonstrating an observed asymptotic convergence order. The final-pair difference is useful evidence but does not by itself establish robust mesh independence. The study is steady-flow and WSS-specific, while the main endpoints also include transient pressure, eddy viscosity and structural stress.

Near-wall spacing differs outside the aneurysm zone between aneurysmal and control meshes. Node-weighted summaries also depend on sampling density; a reversed-node fraction is not a volumetric reverse-flow fraction or a fraction of flow rate.

**Author action:** provide endpoint-specific transient verification, y+ and prism-layer details, solver tolerances/discretization, transition inlet conditions, solid convergence and pressure-transfer checks. If connectivity/areas can be recovered, check whether volume-/area-weighted summaries alter the geometry comparisons. These are requests for evidence, not experiments completed in this revision.

### 10. The pressure convention remains an empirical inference

Original lines 578, 991 and 1325.

The governing RANS equation includes the isotropic turbulent-stress term; the reconstruction omits it because the exported fields satisfy that balance better. That residual check is useful, but better agreement alone does not prove how Fluent/CFD-Post defined the exported pressure. A wrong pressure convention would affect momentum residuals and integration.

**Revision made:** qualify the effective-pressure explanation. **Author action:** verify the exact solver/export variable and pressure convention, including the role of 2 rho k / 3, against the project and version documentation.

### 11. Temporal and structural provenance remain unresolved

Original lines 723, 1202, 1288 and 1485.

The methods select the third cycle, while the results call TAWSS a final-cycle quantity. Only the solver's TAWSS export is retained; the cycle-resolved traction history is unavailable. The slice workbook's averaging interval was inferred from the inlet flow rather than recorded. These cannot be repaired by rewording alone.

The structural model uses ascending-aortic literature constants for all models. The cited Zhu et al. study distinguishes aneurysmal and healthy wall constants and includes prestress, so describe transferring its parameters as an assumption rather than direct validation of this DTAA model. Confirm whether the same wall constants were intentionally used for controls and whether the stated Yeoh form and compressibility settings match the actual implementation.

**Revision made:** explicitly flag the TAWSS averaging cycle and describe the slice profiles as retained descriptive summaries. Author-check comments identify missing structural information.

### 12. Smaller technical and language corrections

- DNS is not a RANS turbulence closure; k-epsilon models unresolved stresses, while DNS resolves turbulent scales.
- TKE per unit mass is a scalar, not the three velocity-fluctuation components.
- A negative plotted WSS component is different from a negative WSS magnitude. The inspected plotting code uses `wss_x`; the caption now identifies the axial component.
- Constant flow rate gives a continuity relation for mean axial velocity; a plotted plane mean of speed is not generally Q/A in recirculating flow.
- Fixed taper provides a common geometry condition; it does not estimate a taper effect.
- With x defined as the downstream axis, both the xy and xz planes contain that axis. Confirm whether the latter should be called an orthogonal longitudinal plane rather than a transverse cross-section; do not silently change anatomy labels in the figure assets.
- Faster flow in a prescribed-flow, unbranched taper does not demonstrate adequate physiological perfusion pressure.
- A training-density change from 1060 to 1050 kg/m3 is small, but unchanged optimization cannot be asserted without retraining.
- Avoid "clear distinction," "it is widely recognized," repeated "this reinforces the need," and paragraph-long descriptions of what figures are "presenting." State the observation and its limit directly.

## What the revision contains

`main_revised.tex` is a separate working copy. It revises 42 original paragraphs/lines and adds source comments for author checks. The original manuscript, displayed equations, result tables, figure assets, author details and bibliography files are preserved. No simulation or reconstruction implementation was changed. `editorial_changes.json` records every paragraph replacement, and `manuscript_changes.diff` shows the full source diff.

The existing generative-AI declaration remains commented, as supplied. This is an author/journal-policy decision, not a visible watermark. If the journal requires disclosure, use wording that accurately covers the actual language, coding and analysis assistance; do not assume a language-only statement is sufficient. No claim is made about removing statistical provenance signals or changing AI-detector scores.

## Writefull and natural academic prose

Writefull remains a reasonable language-editing aid: its official pages describe academic language feedback, paraphrasing and Word/Overleaf support. That does not establish it as a reliable way to remove AI origin or guarantee a detector outcome. In this paper, technical precision should control which suggestions are accepted. The screenshot shows why automatic acceptance is unsuitable.

A useful editing instruction is:

> Edit for clear, concise academic English. Preserve the technical terms, equations, citations, units, numerical values, statistical definitions and level of uncertainty. Remove repetition and generic filler. Flag scientific ambiguities instead of guessing their meaning. Explain any change that could affect interpretation.

Here the objective is specific, evidence-backed prose that the authors can defend. Artificially varying words or adding mistakes to sound human would reduce the quality of the manuscript.

## External sources consulted

- [Writefull official product description](https://www.writefull.com/) - advertised academic editing and paraphrasing capabilities.
- [Writefull paraphrase customization](https://blog.writefull.com/customize-your-paraphrases-with-your-own-prompt-or-preferences/) - protecting technical terms through custom instructions.
- [2022 ACC/AHA aortic disease guideline](https://pmc.ncbi.nlm.nih.gov/articles/PMC9860464/) - aortic-site-specific treatment framing; the full-text endpoint challenged automated access, so no new threshold was inserted into the manuscript.
- [Zhu et al., 2025](https://link.springer.com/article/10.1007/s10237-025-01939-6) and its [institutional full text](https://repository.tudelft.nl/file/File_9f3f6782-d63f-4fbd-9363-e62d0d4886c7) - source and scope of the structural material assumptions.
- [OpenAI official content-provenance documentation](https://developers.openai.com/api/docs/guides/content-provenance) - distinguishes provenance checks from general AI detection; not used as an editing or removal procedure. The user clarified that "watermark" meant generic-sounding prose.

## Verification

The original source SHA-256 was recorded before revision: `aa30a5ec6f4ff99a7756fd00213af29789fa35b132c53832c3b7705636dcd402`.

All 67 distinct citation keys used by the original manuscript exist across its 88 bibliography entries; no duplicate bibliography keys were found. This is a structural check, not proof that each citation supports its associated claim. The revised manuscript compiled successfully with LaTeX/BibTeX to a 50-page PDF with no unresolved references or overfull-box warnings. Visual-QA results and any remaining presentation limitations are recorded in `verification.json` after final inspection.
