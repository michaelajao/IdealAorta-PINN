"""Audit the requested pilot coverage and saved information contracts.

This checks executed records and observation artifacts. It does not turn a short
smoke run into a completed full pilot, or replace independent scientific review.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from observations import array_hash

CONDITIONS = ['point_all_sigma0','point_all_sigma0.02','point_core_sigma0','point_core_sigma0.02',
              'volume_core_sigma0','volume_core_sigma0.02','exclude_near_wall_sigma0',
              'matched_random_removal_sigma0']
METHODS = ['data','cont','scalar_rbf','df_rbf']


def assess_single_draw(root, minimum_steps):
    root=Path(root)
    missing,issues=[],[]
    records=[]
    for case in (1,9):
        for target in (1780,2400):
            unit=root/f'case{case:02d}_t{target}'
            masks=set()
            geometry=unit/'common_geometry_indices.npz'
            if geometry.exists():
                with np.load(geometry) as g:expected_mask=array_hash(g['velocity_evaluation_ids'])
                masks.add(expected_mask)
            else:issues.append(f'{case}/{target}: geometry/scoring indices missing')
            for condition in CONDITIONS:
                observation_hashes=set()
                counts=set()
                obs_path=unit/'observations'/(condition+'.npz')
                meta_path=obs_path.with_suffix('.json')
                if obs_path.exists() and meta_path.exists():
                    with np.load(obs_path) as z:
                        actual_hash=array_hash(*[z[k] for k in ['support_m','weights','values','validation','anchor_ids']])
                        actual_counts=(int((~z['validation']).sum()),int(z['validation'].sum()))
                        if len(np.unique(z['anchor_ids']))!=len(z['anchor_ids']):
                            issues.append(f'{case}/{target}/{condition}: duplicated observation anchors')
                    metadata=json.loads(meta_path.read_text())
                    if actual_hash!=metadata['observation_sha256'] or actual_counts!=(metadata['n_fit_per_time'],metadata['n_val_per_time']):
                        issues.append(f'{case}/{target}/{condition}: saved observation hash/counts disagree with arrays')
                    observation_hashes.add(actual_hash);counts.add(actual_counts)
                else:issues.append(f'{case}/{target}/{condition}: saved observation arrays or metadata missing')
                for method in METHODS:
                    path=unit/'results'/f'{condition}__{method}.json'
                    if not path.exists():
                        missing.append(str(path.relative_to(root)));continue
                    r=json.loads(path.read_text())
                    records.append(r)
                    prefix=f'{case}/{target}/{condition}/{method}'
                    if r['status']!='complete':issues.append(prefix+': not complete')
                    if r['scores']['score_mode']!='full_historical_postprocessing':issues.append(prefix+': no full pressure/WSS evaluation')
                    if not r['clean_targets_unchanged']:issues.append(prefix+': targets changed')
                    if method in ('data','cont') and r['fit']['steps']<minimum_steps:
                        issues.append(prefix+': neural feasibility budget below declared pilot minimum')
                    if not r['oracle_eddy_viscosity_in_pressure']:issues.append(prefix+': oracle pressure label missing')
                    solver=r['scores'].get('pressure_solver',{})
                    if len(solver)!=4 or not all(s.get('converged') for s in solver.values()):
                        issues.append(prefix+': pressure solve convergence is absent or incomplete')
                    boundary=r['scores'].get('known_boundary_fit',{})
                    residuals=boundary.get('wall_rms_over_observed_U',[])+boundary.get('inlet_rms_over_observed_U',[])
                    if len(residuals)!=4 or not np.isfinite(residuals).all() or max(residuals)>.02:
                        issues.append(prefix+': known wall/inlet numerical adequacy is absent or failed')
                    observation_hashes.add(r['observations']['observation_sha256'])
                    counts.add((r['observations']['n_fit_per_time'],r['observations']['n_val_per_time']))
                    masks.add(r['scores']['evaluation_ids_sha256'])
                    if 'shared hold-out retained' not in r['fit']['fit_policy']:
                        issues.append(prefix+': final refitting information differs')
                if len(observation_hashes)>1 or len(counts)>1:
                    issues.append(f'{case}/{target}/{condition}: methods used different observations/splits')
            if len(masks)>1:issues.append(f'{case}/{target}: scoring target set changed across arms')
            obsdir=unit/'observations'
            for sigma in ('0','0.02'):
                point=obsdir/f'point_core_sigma{sigma}.npz'
                volume=obsdir/f'volume_core_sigma{sigma}.npz'
                if point.exists() and volume.exists():
                    with np.load(point) as p,np.load(volume) as v:
                        if not np.array_equal(p['anchor_ids'],v['anchor_ids']) or not np.array_equal(p['validation'],v['validation']):
                            issues.append(f'{case}/{target}: point/volume locations or split differ')
                        if not np.allclose(p['values']-p['clean'],v['values']-v['clean'],atol=1e-14,rtol=1e-12):
                            issues.append(f'{case}/{target}: point/volume noise is not exactly paired')
            left=obsdir/'exclude_near_wall_sigma0.json'
            right=obsdir/'matched_random_removal_sigma0.json'
            if left.exists() and right.exists():
                a=json.loads(left.read_text());b=json.loads(right.read_text())
                if any(a[k]!=b[k] for k in ['n_fit_per_time','n_val_per_time']):
                    issues.append(f'{case}/{target}: removal arms do not match both split counts')
    return {'status':'draw_coverage_and_contracts_pass' if not missing and not issues else 'incomplete_or_failed',
            'expected_results':128,'completed_records':len(records),'minimum_neural_steps':minimum_steps,
            'missing_results':missing,'issues':issues,
            'does_not_prove':['native CAD equivalence of reconstructed geometry','clinical validity',
                              'independence of current cases','complete convergence/capacity assessment']}


def assess(root,stage='development'):
    root=Path(root)
    if stage=='feasibility':
        report=assess_single_draw(root,2000)
        report['stage']='bounded_feasibility_only'
        return report
    reports={}
    issues=[]
    for draw in (0,1,2):
        directory=root/f'draw{draw:02d}'
        reports[str(draw)]=assess_single_draw(directory,20000)
        manifest=directory/'execution_protocol.json'
        if not manifest.exists():
            issues.append(f'draw{draw:02d}: execution protocol missing')
        else:
            config=json.loads(manifest.read_text())['config']
            for key,expected in [('grid_seed',draw),('noise_seed',22001+draw),('init_seed',draw)]:
                if config.get(key)!=expected:
                    issues.append(f'draw{draw:02d}: {key} differs from declared seed')
    passed=not issues and all(r['status']=='draw_coverage_and_contracts_pass' for r in reports.values())
    return {'status':'development_pilot_coverage_and_contracts_pass' if passed else 'incomplete_or_failed',
            'stage':'full_exploratory_development_pilot','expected_results':384,
            'completed_records':sum(r['completed_records'] for r in reports.values()),
            'issues':issues,'draws':reports,
            'remaining_scientific_audit':'capacity/convergence evidence, geometry uncertainty and interpretation still require review'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--stage',choices=['feasibility','development'],default='development')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=assess(args.root,args.stage)
    text=json.dumps(report,indent=2)
    if args.output:args.output.write_text(text,encoding='utf-8')
    print(text)
    raise SystemExit(0 if report['status'] in ['development_pilot_coverage_and_contracts_pass',
                                            'draw_coverage_and_contracts_pass'] else 1)
