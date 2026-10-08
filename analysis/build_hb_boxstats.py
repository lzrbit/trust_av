#!/usr/bin/env python3
"""Export 15 descriptive HB braking boxes from participant-first clip means.

No model, confidence interval, bootstrap or significance test is recomputed.
Required packages: numpy, pandas, pyarrow. Original inputs are read-only.
"""
from settings import WORKSPACE as ROOT, DATA_DIR, QUESTIONNAIRE, REPO, relative_label
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    root = parser.parse_args().project_root.resolve()
    dest = root / 'outputs/boxes'
    tables, reports = dest / 'tables', dest / 'reports'
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    sources = {
        'windows': DATA_DIR / 'windows.parquet',
        'design': REPO / 'data/event_design.csv',
        'profile_reference': root / 'outputs/parameters/tables/controlled_parameter_clip_profiles.csv',
        'prior_audit': root / 'outputs/parameters/reports/controlled_parameter_analysis_audit.json',
    }
    initial_hashes = {relative_label(path,root): sha(path) for path in sources.values()}
    prior = json.loads(sources['prior_audit'].read_text())
    for name in ['windows', 'design']:
        assert sha(sources[name]) == prior['source_hashes'][name]['sha256'], ('Changed analysis input', name)
    rel = str(sources['profile_reference'].relative_to(root))
    assert sha(sources['profile_reference']) == prior['output_hashes'][rel]

    cols = ['participant_id', 'scenario', 'event_key', 'window_index', 'window_row_id', 'risk_raw', 'risk']
    windows = pd.read_parquet(sources['windows'], columns=cols)
    assert windows.window_row_id.is_unique
    assert not windows.duplicated(['participant_id', 'event_key', 'window_index']).any()
    assert len(windows) == 181776 and windows.participant_id.nunique() == 2164
    assert windows.risk_raw.eq(0).sum() == 1810 and windows.risk.notna().sum() == 179966
    np.testing.assert_allclose(windows.risk, windows.risk_raw.where(windows.risk_raw > 0), equal_nan=True)
    assert windows.risk.dropna().between(1, 10).all()
    design = pd.read_csv(sources['design'])
    assert not design.duplicated(['scenario', 'event_key']).any()
    hb_design = design[design.display_scenario.eq('HB')]
    assert len(hb_design) == 27 and hb_design.scenario.eq('B3').all()
    assert hb_design.groupby('design_braking_m_s2').size().to_dict() == {-8.: 9, -5.: 9, -2.: 9}
    hb = windows[windows.scenario.eq('B3')].merge(
        hb_design[['scenario', 'event_key', 'design_braking_m_s2']],
        on=['scenario', 'event_key'], how='left', validate='many_to_one')
    assert hb.design_braking_m_s2.notna().all() and hb.window_index.between(1, 5).all()
    assert len(hb) == 2164 * 4 * 5
    person = hb.groupby(['participant_id', 'design_braking_m_s2', 'window_index']).risk.agg(['mean', 'count']).reset_index()
    person = person[person['count'].gt(0)]
    assert person['mean'].between(1, 10).all() and person['count'].between(1, 4).all()
    assert not person.duplicated(['participant_id', 'design_braking_m_s2', 'window_index']).any()
    reference = pd.read_csv(sources['profile_reference'], float_precision='round_trip')
    reference = reference[reference.analysis_variant.eq('assigned_design_primary') &
                          reference.display_scenario.eq('HB') & reference.factor.eq('design_braking')].copy()
    reference['design_braking_m_s2'] = reference.level.astype(float)
    assert len(reference) == 15
    stats, flier_rows, checks = [], [], []
    for clip in range(1, 6):
        for level in [-2, -5, -8]:
            values = np.sort(person.loc[person.window_index.eq(clip) &
                                       person.design_braking_m_s2.eq(level), 'mean'].to_numpy(float))
            assert len(values) > 0 and np.isfinite(values).all()
            q1, median, q3 = np.quantile(values, [.25, .5, .75], method='linear')
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            inside = values[(values >= lower) & (values <= upper)]
            assert len(inside) > 0
            whisker_low, whisker_high = float(inside[0]), float(inside[-1])
            fliers = values[(values < whisker_low) | (values > whisker_high)]
            box_id = f'HB_braking_{level}_clip_{clip}'
            cell = hb[hb.window_index.eq(clip) & hb.design_braking_m_s2.eq(level)]
            mean = float(values.mean())
            old = reference[reference['clip'].eq(clip) & reference.design_braking_m_s2.eq(level)].iloc[0]
            assert len(values) == old.n_participants
            assert int(cell.risk.notna().sum()) == old.n_valid_ratings
            error = abs(mean - float(old['mean']))
            assert error < 1e-12, (box_id, error)
            assert whisker_low <= q1 <= median <= q3 <= whisker_high
            assert 1 <= whisker_low <= whisker_high <= 10
            row = dict(box_id=box_id, scenario='B3', display_scenario='HB', clip=clip,
                       design_braking_m_s2=level, n_participants=len(values),
                       n_valid_ratings=int(cell.risk.notna().sum()),
                       n_assigned_rating_opportunities=len(cell), n_no_operation=int(cell.risk_raw.eq(0).sum()),
                       n_distinct_design_events=cell.event_key.nunique(),
                       mean=mean, q1=float(q1), median=float(median), q3=float(q3), iqr=float(iqr),
                       lower_fence=float(lower), upper_fence=float(upper),
                       whisker_low=whisker_low, whisker_high=whisker_high,
                       n_fliers=len(fliers), fliers_json=json.dumps(fliers.tolist(), separators=(',', ':')))
            stats.append(row)
            for index, value in enumerate(fliers, start=1):
                flier_rows.append(dict(box_id=box_id, clip=clip, design_braking_m_s2=level,
                                       flier_index=index, participant_mean_risk=float(value),
                                       side='below' if value < whisker_low else 'above'))
            checks.append(dict(box_id=box_id, n_participants=len(values), v5_n_participants=int(old.n_participants),
                               mean=mean, v5_mean=float(old['mean']), absolute_mean_difference=error,
                               n_valid_ratings=int(cell.risk.notna().sum()), v5_n_valid_ratings=int(old.n_valid_ratings)))
    stats_frame = pd.DataFrame(stats)
    assert len(stats_frame) == 15 and stats_frame.n_participants.min() == 1749 and stats_frame.n_participants.max() == 1787
    assert stats_frame.n_valid_ratings.sum() == hb.risk.notna().sum()
    assert stats_frame.n_no_operation.sum() == hb.risk_raw.eq(0).sum()
    assert stats_frame.n_fliers.sum() == len(flier_rows)
    stats_path = tables / 'figure_1_hb_braking_boxstats.csv'
    flier_path = tables / 'figure_1_hb_braking_fliers.csv'
    stats_frame.to_csv(stats_path, index=False, float_format='%.17g')
    pd.DataFrame(flier_rows, columns=['box_id', 'clip', 'design_braking_m_s2', 'flier_index',
                                    'participant_mean_risk', 'side']).to_csv(flier_path, index=False, float_format='%.17g')
    saved = pd.read_csv(stats_path, float_precision='round_trip')
    assert len(saved) == 15
    assert sum(len(json.loads(s)) for s in saved.fliers_json) == len(flier_rows)
    assert all(sha(Path(rel) if Path(rel).is_absolute() else root / rel) == digest for rel, digest in initial_hashes.items())
    audit = dict(status='pass', created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Descriptive box statistics only; no fitted model, bootstrap, confidence interval or significance test.',
        scenario='HB', internal_scenario='B3', design_levels_m_s2=[-2,-5,-8], clips=[1,2,3,4,5],
        boxes=15, participant_n_range=[int(saved.n_participants.min()),int(saved.n_participants.max())],
        source_cohort_participants=2164, source_country_count=29,
        HB_assigned_ratings=len(hb), HB_valid_ratings=int(hb.risk.notna().sum()), HB_no_operation=int(hb.risk_raw.eq(0).sum()),
        mean_match_to_v5_max_abs_error=max(x['absolute_mean_difference'] for x in checks),
        all_group_counts_match_v5=True, total_flier_points=len(flier_rows),
        estimand='Within each participant x design braking level x clip, average operated event ratings; boxes summarise the empirical distribution of these participant means, each participant once per box.',
        quantile_method='numpy.quantile method=linear; Q1=.25, median=.5, Q3=.75',
        whisker_rule='Most extreme observed participant mean within [Q1-1.5IQR,Q3+1.5IQR]; no percentile whiskers and no autorange override.',
        flier_rule='Every participant mean beyond observed whisker endpoints is exported, with duplicate values preserved. Fliers remain in all calculations. Any plot-only omission must be declared by its caption.',
        zero_rule='Only video risk 0 is no operation and missing; attitude scores are neither read nor transformed.',
        interpretation='A person may appear in multiple boxes. Marginal observed distributions do not adjust for other design factors. They are not confidence intervals or causal effects. Existing paired contrasts remain separate estimates.',
        HB_acceleration_provenance='Group uses the design value confirmed by velocity-derived braking; it does not replace known stored ax_n discrepancies for HB events7-9.',
        source_hashes=initial_hashes, original_inputs_unchanged=True, group_checks=checks,
        output_hashes={str(path.relative_to(root)):sha(path) for path in [stats_path, flier_path]},
        script_sha256=sha(Path(__file__)), runtime=dict(numpy=np.__version__, pandas=pd.__version__))
    report_path = reports / 'hb_braking_boxstats_validation.json'
    report_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:audit[k] for k in ['status','boxes','participant_n_range','mean_match_to_v5_max_abs_error',
                                        'HB_valid_ratings','HB_no_operation','total_flier_points']}, indent=2))


if __name__ == '__main__':
    main()
