"""Audited native development results; dependent repeats stay within geometry."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

METRICS={
    'velocity_relative_L2':('velocity','vel_rel_l2'),
    'velocity_vector_RMSE_m_s':('velocity','vel_rms_ms'),
    'pressure_relative_L2_oracle':('pressure','unsteady+mut','p_rel_l2'),
    'pressure_RMSE_Pa_oracle':('pressure','unsteady+mut','p_rms_Pa'),
    'WSS_magnitude_relative_L2':('wss','wss_mag_rel_l2'),
    'WSS_vector_relative_L2':('wss','wss_vec_rel_l2'),
    'WSS_magnitude_RMSE_Pa':('wss','wss_mag_rms_Pa'),
    'WSS_vector_RMSE_Pa':('wss','wss_vec_rms_Pa'),
    'acceleration_vector_RMSE_m_s2':('acceleration_rms_error_m_s2',),
    'divergence_RMS_per_s_10um':('finite_difference_divergence_rms_per_s','1e-05'),
    'cap_imbalance':('cap_flux_balance','relative_cap_imbalance'),
}


def main():
    root=Path(sys.argv[1]).resolve()
    sys.path[:0]=[str(Path('/home/olarinoyem/measurement_realism_2026_10_22/code_v22')),
                  '/home/olarinoyem/aorta_rescore']
    from assess_pilot import assess
    audit=assess(root,'development')
    assert audit['status']=='development_pilot_coverage_and_contracts_pass' and audit['completed_records']==384
    rows=[]
    for path in sorted(root.glob('draw*/case*/results/*.json')):
        value=json.loads(path.read_text())
        row={key:value[key] for key in ('case','target_ms','condition','method')}
        row.update(draw=int(path.parents[2].name[4:]),seconds=value['seconds'],
                   result_file=str(path.relative_to(root)),result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for metric,keys in METRICS.items():
            item=value['scores']
            for key in keys:item=item[key]
            row[metric]=item
        rows.append(row)
    assert len(rows)==384 and {r['case'] for r in rows}=={1,9}
    destination=root/'summary';destination.mkdir()
    with (destination/'all_case_phase_draw_results.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    grouped={}
    for row in rows:grouped.setdefault((row['case'],row['condition'],row['method']),[]).append(row)
    case_rows=[]
    for (case,condition,method),values in sorted(grouped.items()):
        assert len(values)==6
        case_rows.append({'case':case,'condition':condition,'method':method,
            'dependent_phase_draw_repeats':len(values),
            **{metric:statistics.mean(v[metric] for v in values) for metric in METRICS}})
    with (destination/'case_means_over_phases_and_draws.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(case_rows[0]));writer.writeheader();writer.writerows(case_rows)
    indexed={(r['case'],r['target_ms'],r['draw'],r['condition'],r['method']):r for r in rows}
    contrasts=[]
    for row in rows:
        condition=row['condition'];pairs=[]
        if condition=='exclude_near_wall_sigma0':pairs.append(('near_wall_location_minus_matched_count','matched_random_removal_sigma0'))
        if condition.startswith('volume_core_'):pairs.append(('volume_average_minus_matched_point',condition.replace('volume_core_','point_core_')))
        if condition.endswith('sigma0.02'):pairs.append(('noise_0.02_minus_no_noise',condition[:-len('sigma0.02')]+'sigma0'))
        for kind,reference in pairs:
            other=indexed[(row['case'],row['target_ms'],row['draw'],reference,row['method'])]
            contrasts.append({k:row[k] for k in ('case','target_ms','draw','method')}|
                {'contrast':kind,'condition':condition,'reference_condition':reference}|
                {metric:row[metric]-other[metric] for metric in METRICS})
    with (destination/'paired_condition_differences.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(contrasts[0]));writer.writeheader();writer.writerows(contrasts)
    receipt={'status':'audited_development_tables_complete','records':384,'geometry_units':2,
        'geometry_ids':[1,9],'scope':'exploratory existing-development cases; no independent evaluation',
        'aggregation':'six dependent phase/draw repeats averaged WITHIN each geometry; no population significance claim',
        'contrast_sign':'condition error minus reference error; positive means larger error, except cap imbalance remains a diagnostic',
        'primary_pressure':'unsteady momentum with supplied CFD eddy viscosity; oracle closure',
        'WSS_absolute_error':'same aneurysm-zone nodes and estimator as historical relative errors',
        'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'tables':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.glob('*.csv')}}
    (destination/'summary_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
