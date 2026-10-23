"""Write bounded audit metadata without modifying live scientific sources."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPORT = ROOT / 'report/measurement_realism_2026_10_22'
EXPERIMENT = ROOT / 'experiments/measurement_realism_2026_10_22'
now = datetime.datetime.now(datetime.timezone.utc).isoformat()

def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')

old_access = REPORT / 'prior_access_declaration_20261023.json'
if not (HERE / 'prior_access_declaration_original.json').exists():
    shutil.copyfile(old_access, HERE / 'prior_access_declaration_original.json')
access = json.loads((HERE / 'prior_access_declaration_original.json').read_text(encoding='utf-8'))
access['recorded_utc'] = now
access['supersedes'] = 'prior_access_declaration_original.json; original retained, including its overbroad wording'
access['allowed_prior_access'] = (
    'Publication, release manifests and bounded metadata inspection; also scalar '
    'velocity-component maxima from aorta03_HR.h5, related to the proposed '
    'aorta03trans evaluation anatomy. No blanket claim that no velocity values were inspected.'
)
access['evaluation_related_velocity_summary_access'] = {
    'file': 'aorta03_HR.h5',
    'accessed_utc': '2026-10-22T10:04:25.362Z',
    'datasets_and_stored_values': {
        'u_max': 0.9341978430747986,
        'v_max': 0.5550472736358643,
        'w_max': 0.6191098690032959,
    },
    'units': 'stored scalar values; not independently reinterpreted as units here',
    'scope': 'summary datasets read through bounded HTTP ranges; full velocity arrays not read by that inspection',
    'bytes_downloaded': 1432128,
    'http_range_requests': 22,
    'relation': 'related single-frame anatomy; not the proposed aorta03trans_HR.h5 temporal file',
    'evidence': 'retained tool output, 2026-10-22T10:04:25.362Z, session 01a16311-ab3e-76b0-88e2-a9ae4beb8836',
    'implication': 'disclose prior summary access; do not describe related anatomy as wholly unseen',
}
access['future_rule'] = 'Goal paused; no evaluation access or scheduling until design revision and a new source/design seal incorporating this correction'
write(HERE / 'prior_access_declaration_corrected.json', access)
write(old_access, access)

progress_path = REPORT / 'progress.json'
progress = json.loads(progress_path.read_text(encoding='utf-8'))
if not (REPORT / 'progress_before_user_audit_pause_20261023.json').exists():
    shutil.copyfile(progress_path, REPORT / 'progress_before_user_audit_pause_20261023.json')
progress.update(updated_utc=now, goal_status='paused at explicit user request; incomplete',
    progress_classification='audit checkpoint; automatic pilot launch disabled; scientific design revision pending',
    live_processes=[{'job':'capacity_v21','supervisor_pid':1780664,'child_pid':1780672,
        'last_verified':'2026-10-23T11:38:03.055207+00:00',
        'scope':'unchanged existing capacity sweep only; no automatic full pilot launch'}],
    remaining=['author decisions and complete red manuscript review',
        'revise scientific design using supplied audit before any new scheduling',
        'full development comparison and independent protocol seal remain incomplete'],
    queued_native_pipeline={'pid':1791337,'status':'stopped_at_user_request_before_any_stage',
        'automatic_launch_enabled':False,'receipt':'../../doc/revision_audit_2026-10-23/brosnan_pause_receipt.json',
        'current_feasibility_records':0,'current_final_development_records':0},
    thread_resource_development={'status':'completed resource probes; no new benchmarks launched during pause',
        'native_factor_and_solve_4_to_8_elapsed_ratio':1.566966164,
        'scope':'one shared-machine candidate, not a full-pilot speedup'},
    external_prior_access_correction='../../doc/revision_audit_2026-10-23/prior_access_declaration_corrected.json')
progress['v21_partial_capacity'] = {
    'unit':'case01_t1780', 'scope':'partial observed-data capacity; not completed method comparison',
    'point_scalar':'settled within the declared test',
    'point_df':{'status':'capacity_unsettled','validation_change':0.011563706326761269,
                'prediction_change':0.03143311149816512,'tolerance':0.02},
    'volume_scalar':{'status':'capacity_unsettled','validation_change':0.04270866895572418,
                    'prediction_change':0.013494970347356404,'tolerance':0.02},
}
progress['resume_plan'] = 'Old queue superseded by user pause; revise design and obtain explicit resume before launching'
progress['external_adapter']['final_evaluation_seal'] = 'not created; design requires revision and corrected access declaration'
progress['external_adapter']['evaluation_related_summary_values_accessed'] = True
write(progress_path, progress)
write(HERE / 'progress_at_pause.json', progress)
shutil.copyfile(REPORT / 'checks/remote_v22_20261023/integration_tests_v22_threads8_receipt.json',
                HERE / 'integration_tests_brosnan_66_receipt.json')
write(HERE / 'preservation_manifest.json', {
    'created_utc': now,
    'files': {p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in HERE.iterdir() if p.is_file() and p.name != 'preservation_manifest.json'},
    'scientific_sources_sha256': {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in EXPERIMENT.glob('*.py')},
    'remote_evidence_root':'/home/olarinoyem/measurement_realism_2026_10_22',
    'large_reports_policy':'full fit/model evidence remains in dated local and remote report roots; compact pause/test/access receipts are versioned here',
})
print('Audit checkpoint metadata written; scientific Python sources unchanged.')
