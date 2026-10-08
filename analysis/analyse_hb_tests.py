#!/usr/bin/env python3
"""Post hoc paired-mean annotation tests for the existing HB braking boxes.

Reads protected participant-level data; exports aggregate statistics only.
Dependencies: numpy, pandas, pyarrow, scipy. No models or bootstrap are fitted.
"""
from pathlib import Path
from settings import WORKSPACE as ROOT, DATA_DIR, REPO
from datetime import datetime, timezone
import argparse, hashlib, json
import numpy as np
import pandas as pd
from scipy import stats


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    result = np.empty_like(p)
    result[order] = adjusted
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', type=Path, default=ROOT)
    ap.add_argument('--data-dir', type=Path, help='Directory containing windows.parquet')
    ap.add_argument('--design-csv', type=Path)
    ap.add_argument('--output-dir', type=Path)
    args = ap.parse_args(); root = args.project_root.resolve()
    data = (args.data_dir or DATA_DIR).resolve()
    design_path = (args.design_csv or REPO/'data/event_design.csv').resolve()
    out = (args.output_dir or root/'outputs/hb_tests').resolve()
    for folder in ['tables', 'reports']:
        (out/folder).mkdir(parents=True, exist_ok=True)
    sources = {'windows': data/'windows.parquet', 'event_design': design_path}
    source_hashes = {name: {'file': p.name, 'sha256': sha(p)} for name, p in sources.items()}
    plan_path = out/'reports/hb_braking_significance_plan.json'
    specification = {
        'scope': 'Post hoc annotation of an existing descriptive figure, not preregistered; previous HB results were known.',
        'population': 'All retained participants with operated responses at both compared levels in the clip; no country threshold.',
        'aggregation': 'Mean of available operated event ratings within participant x HB design braking level x clip, then paired stronger-minus-weaker level difference.',
        'contrasts': [[-5, -2], [-8, -2], [-8, -5]], 'clips': [1, 2, 3, 4, 5],
        'test': 'Two-sided one-sample Student t test of participant-level paired differences against zero; df=Npaired-1.',
        'multiplicity': 'Holm step-down correction across all 15 pair-by-clip tests as one new post hoc family.',
        'CI': 'Pointwise 95% Student t confidence interval for each paired mean difference; not Holm simultaneous intervals.',
        'stars': {'***': 'Holm p < .001', '**': 'Holm p < .01', '*': 'Holm p < .05', 'ns': 'Holm p >= .05'},
        'interpretation': 'Stars concern paired means, not box medians or correlation; marginal over other design factors, not causal effects.',
        'existing_analysis': 'Existing 75-primary-contrast estimates and bootstrap intervals remain unchanged and form a separate family.',
        'zero_rule': 'Only video-risk 0 is no operation and omitted before within-person aggregation.',
        'sources': source_hashes,
    }
    if plan_path.exists():
        assert json.loads(plan_path.read_text())['specification'] == specification, 'Plan/input changed'
    else:
        plan_path.write_text(json.dumps({'frozen_before_test_execution_utc': datetime.now(timezone.utc).isoformat(), 'specification': specification}, indent=2)+'\n')
    # Plan is on disk before outcome differences or test statistics are computed.
    w = pd.read_parquet(sources['windows'], columns=['participant_id', 'scenario', 'event_key', 'window_index', 'risk_raw', 'risk'])
    assert not w.duplicated(['participant_id', 'event_key', 'window_index']).any()
    np.testing.assert_allclose(w.risk, w.risk_raw.where(w.risk_raw > 0), equal_nan=True)
    design = pd.read_csv(design_path)
    d = design[design.display_scenario.eq('HB')][['scenario', 'event_key', 'design_braking_m_s2']]
    assert len(d) == 27 and not d.duplicated(['scenario', 'event_key']).any()
    hb = w[w.scenario.eq('B3')].merge(d, on=['scenario', 'event_key'], how='left', validate='many_to_one')
    assert hb.design_braking_m_s2.notna().all()
    person = hb.groupby(['participant_id', 'window_index', 'design_braking_m_s2']).risk.mean().unstack('design_braking_m_s2')
    rows = []
    for clip in specification['clips']:
        one = person.xs(clip, level='window_index')
        for stronger, weaker in specification['contrasts']:
            paired = one[[stronger, weaker]].dropna()
            delta = (paired[stronger] - paired[weaker]).to_numpy(float)
            n = len(delta); assert n >= 2
            mean = float(delta.mean()); sd = float(delta.std(ddof=1)); se = sd/np.sqrt(n)
            assert se > 0, 'Degenerate paired differences require explicit review'
            t = mean/se; p = float(2*stats.t.sf(abs(t), n-1)); half = float(stats.t.ppf(.975, n-1)*se)
            independent = stats.ttest_rel(paired[stronger], paired[weaker])
            np.testing.assert_allclose([t, p], [independent.statistic, independent.pvalue], rtol=1e-12, atol=1e-14)
            rows.append({'scenario': 'B3', 'display_scenario': 'HB', 'clip': clip,
                'stronger_braking_m_s2': stronger, 'weaker_braking_m_s2': weaker, 'n_paired': n,
                'mean_stronger_paired': float(paired[stronger].mean()), 'mean_weaker_paired': float(paired[weaker].mean()),
                'mean_difference': mean, 'sd_paired_difference': sd, 'se_paired_difference': float(se),
                'ci95_pointwise_low': mean-half, 'ci95_pointwise_high': mean+half,
                't_statistic': float(t), 'df': n-1, 'p_two_sided': p})
    result = pd.DataFrame(rows); result['p_holm_15'] = holm(result.p_two_sided)
    result['significance'] = result.p_holm_15.map(lambda p: '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns')
    result['family_size'] = 15; result['analysis_status'] = 'post_hoc_figure_annotation'
    assert len(result) == 15 and not result.duplicated(['clip', 'stronger_braking_m_s2', 'weaker_braking_m_s2']).any()
    assert np.isfinite(result.select_dtypes(include=np.number)).all().all()
    path = out/'tables/hb_braking_paired_tests.csv'; result.to_csv(path, index=False, float_format='%.17g')
    # Compare the five unchanged extreme-level point estimates with the existing family.
    checks = []
    reference = root/'outputs/parameters/tables/controlled_parameter_paired_contrasts.csv'
    if reference.is_file():
        old = pd.read_csv(reference)
        old = old[old.analysis_variant.eq('assigned_design_primary') & old.display_scenario.eq('HB') & old.factor.eq('design_braking')]
        assert len(old) == 5
        for row in result[(result.stronger_braking_m_s2 == -8) & (result.weaker_braking_m_s2 == -2)].itertuples():
            previous = old[old['clip'].eq(row.clip)].iloc[0]
            err = abs(row.mean_difference - previous.mean_difference); assert err < 1e-12
            checks.append({'clip': row.clip, 'absolute_mean_difference_error': err})
    assert all(sha(p) == source_hashes[k]['sha256'] for k, p in sources.items())
    audit = {'status': 'pass', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': sha(plan_path), 'script_sha256': sha(Path(__file__)), 'sources': source_hashes,
        'output': {'file': path.name, 'sha256': sha(path), 'rows': len(result)},
        'n_paired_range': [int(result.n_paired.min()), int(result.n_paired.max())],
        'significance_counts': result.significance.value_counts().to_dict(),
        'matched_frozen_extreme_point_estimates': checks,
        'participant_records_exported': False, 'models_refitted': False, 'bootstrap_recomputed': False,
        'original_inputs_unchanged': True, 'CI_family_note': specification['CI'],
        'test_family_note': specification['multiplicity']}
    (out/'reports/hb_braking_significance_validation.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    print(result[['clip', 'stronger_braking_m_s2', 'weaker_braking_m_s2', 'n_paired', 'mean_difference', 'p_holm_15', 'significance']].to_string(index=False))
    print(json.dumps({k: audit[k] for k in ['status', 'n_paired_range', 'significance_counts']}))


if __name__ == '__main__':
    main()
