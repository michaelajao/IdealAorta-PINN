# Manuscript generators

Run these scripts from the repository root using the environment in `requirements.txt`.
They read retained CFD data, checkpoints and reconstruction records; they do not train
models. Paths are resolved from the script location, so another working directory also works.

Figures and LaTeX tables remain under `paper/revision_2026_09/manuscript/`.
CSV/JSON summaries go to `report/tables/`; field-map arrays go to `tmp/`.
The manuscript and study data are intentionally excluded from Git.

| Command | Outputs |
| --- | --- |
| `python scripts/manuscript/generate_bc_xwss.py` | Boundary waveforms and signed x-WSS lines |
| `python scripts/manuscript/generate_cfd_summary.py` | Common-time CFD summary and inflow groups |
| `python scripts/manuscript/generate_cfd_views.py [sl wss]` | Streamlines and wall-shear surface views; defaults to both |
| `python scripts/manuscript/generate_evidence.py` | Reconstruction tables, figures and evidence audit |
| `python scripts/manuscript/generate_field_maps.py [case target_ms]` | Pressure/WSS maps; defaults to Case 4 at 1780 ms |
| `python scripts/manuscript/generate_plane_profiles.py` | Plane-wise speed and turbulent kinetic energy |
| `python scripts/manuscript/generate_reference_table.py` | Complete-CFD post-processing reference table |
| `python scripts/manuscript/generate_robustness_table.py` | Robustness table and audit; requires all 14 case-phase units |
| `python scripts/manuscript/generate_wss_montage.py` | Systolic and diastolic WSS montages |

For the adverse diastolic field-map example, use
`python scripts/manuscript/generate_field_maps.py 7 2400`.

While robustness jobs are still running,
`python scripts/manuscript/generate_robustness_table.py --completed-groups`
writes `tables/robustness_completed.tex` and an interim JSON audit. Each
numerical row requires all seven cases for that method and phase; incomplete
groups are explicitly marked pending. The default command still requires all
14 case-phase units for every method.

These scripts were moved out of the manuscript directory. Use the commands above;
the old `paper/revision_2026_09/manuscript/generate_*.py` entry points were removed.
