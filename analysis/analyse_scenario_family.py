#!/usr/bin/env python3
"""Scenario-family decomposition of the differential trust-rating contrast and
descriptive classification of ACC/LKS change directions (manuscript v9, SI S11,
Supplementary Tables S21-S22).

Frozen specification: config/scenario_family.json. Reuses saved quantities only:
recorded pre/post ratings, clipwise participant profiles and the five-fold
cross-fitted high-point positions. No model is retrained and no high point is
reselected. Stage prerequisites: dynamics (a2_* tables) and differential-trust
(within_person_trust_contrast_risk3.csv).
"""
from settings import WORKSPACE as ROOT, DATA_DIR, REPO, relative_label
from analyse_dynamics import SEED as SEED_FOLDS, BLOCKS, NPOS, profile_matrix
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, platform
import numpy as np, pandas as pd, patsy, scipy, statsmodels, statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

PRIOR = ['trust_pre_overall', 'trust_pre_acc', 'trust_pre_lks',
         'accept_pre_delegate', 'accept_pre_monitor_raw', 'accept_pre_distract']
FAMILY = ['mean_lat', 'mean_long', 'excursion_lat', 'excursion_long', 'late_lat', 'late_long']
RISK7 = FAMILY + ['positive_fraction']
POOLED = ['mean_risk', 'excursion', 'late_change', 'positive_fraction']
PLAN_PATH = REPO / 'config/scenario_family.json'
POOLED_PLAN_PATH = REPO / 'config/trust_contrast.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def hc1_fit(y, design):
    return sm.OLS(np.asarray(y, float), design).fit(cov_type='HC1', use_t=False)


def coef_rows(fit, design, outcome, family_terms):
    ci = fit.conf_int(alpha=.05)
    rows = []
    for t in design.columns:
        rows.append({'term': t, 'estimate': float(fit.params[t]), 'se_HC1': float(fit.bse[t]),
                     'ci_low': float(ci.loc[t, 0]), 'ci_high': float(ci.loc[t, 1]),
                     'p_two_sided_normal': float(fit.pvalues[t]),
                     'role': 'risk_family' if t in family_terms else 'adjustment',
                     'n': int(fit.nobs), 'outcome': outcome})
    return pd.DataFrame(rows)


def wald_equal(fit, a, b):
    c = np.zeros(len(fit.params))
    c[fit.params.index.get_loc(a)] = 1
    c[fit.params.index.get_loc(b)] = -1
    est = float(c @ fit.params.to_numpy())
    var = float(c @ fit.cov_params().to_numpy() @ c)
    z = est / np.sqrt(var)
    return {'contrast': f'{a} - {b}', 'estimate': est, 'se_HC1': float(np.sqrt(var)),
            'ci_low': est - 1.959964 * np.sqrt(var), 'ci_high': est + 1.959964 * np.sqrt(var),
            'p_two_sided_normal': float(2 * scipy.stats.norm.sf(abs(z)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', type=Path, default=ROOT)
    args = ap.parse_args()
    root = args.project_root.resolve()
    out = root / 'outputs/parameters'
    reports, tables = out / 'reports', out / 'tables'
    reports.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    plan = json.loads(PLAN_PATH.read_text())
    pooled_plan = json.loads(POOLED_PLAN_PATH.read_text())
    lat, long_ = plan['lateral_family'], plan['longitudinal_family']
    draws, seed_boot = plan['direction_bootstrap']['draws'], plan['direction_bootstrap']['seed']

    sources = {'participants': DATA_DIR / 'participants.parquet', 'windows': DATA_DIR / 'windows.parquet',
               'features': root / 'outputs/core/tables/a2_participant_risk_features.csv',
               'peaks': root / 'outputs/core/tables/a2_crossfit_peak_locations.csv',
               'pooled_risk3': tables / 'within_person_trust_contrast_risk3.csv'}
    source_sha = {relative_label(p, root): sha(p) for p in sources.values()}
    p = pd.read_parquet(sources['participants'])
    w = pd.read_parquet(sources['windows'])
    saved = pd.read_csv(sources['features'])
    peaks_saved = pd.read_csv(sources['peaks'])
    pooled_saved = pd.read_csv(sources['pooled_risk3'])
    assert p.participant_id.is_unique

    # ---------------- Part A: direction classification ----------------
    da = (p.trust_post_acc - p.trust_pre_acc).to_numpy(float)
    dl = (p.trust_post_lks - p.trust_pre_lks).to_numpy(float)
    assert np.isfinite(da).all() and np.isfinite(dl).all()
    cats = {'acc_down_lks_up': (da < 0) & (dl > 0), 'acc_up_lks_down': (da > 0) & (dl < 0),
            'both_up': (da > 0) & (dl > 0), 'both_down': (da < 0) & (dl < 0),
            'acc_zero_lks_nonzero': (da == 0) & (dl != 0), 'lks_zero_acc_nonzero': (dl == 0) & (da != 0),
            'both_zero': (da == 0) & (dl == 0)}
    assert list(cats) == plan['direction_categories']
    M = np.column_stack([v.astype(float) for v in cats.values()])
    assert (M.sum(axis=1) == 1).all()
    rng = np.random.default_rng(seed_boot)
    n = len(da)
    W = rng.multinomial(n, np.full(n, 1 / n), size=draws).astype(float)
    boot = (W @ M) / n
    rows = []
    for j, (k, v) in enumerate(cats.items()):
        rows.append({'category': k, 'n': int(v.sum()), 'proportion': float(v.mean()),
                     'ci_low': float(np.percentile(boot[:, j], 2.5)), 'ci_high': float(np.percentile(boot[:, j], 97.5)),
                     'n_participants': n, 'bootstrap_draws': draws, 'seed': seed_boot})
    opp = cats['acc_down_lks_up'] | cats['acc_up_lks_down']
    same = cats['both_up'] | cats['both_down']
    for k, v in [('opposite_direction_any', opp), ('same_direction_any', same),
                 ('acc_changed', da != 0), ('lks_changed', dl != 0), ('either_changed', (da != 0) | (dl != 0))]:
        bb = (W @ v.astype(float)) / n
        rows.append({'category': k, 'n': int(v.sum()), 'proportion': float(v.mean()),
                     'ci_low': float(np.percentile(bb, 2.5)), 'ci_high': float(np.percentile(bb, 97.5)),
                     'n_participants': n, 'bootstrap_draws': draws, 'seed': seed_boot})
    direction = pd.DataFrame(rows)
    direction.to_csv(tables / 'scenario_family_direction_classification.csv', index=False, float_format='%.17g')
    rel = lambda x, y: np.where(x > y, 'lks_above', np.where(x < y, 'acc_above', 'equal'))
    pre_rel = rel(p.trust_pre_lks.to_numpy(float), p.trust_pre_acc.to_numpy(float))
    post_rel = rel(p.trust_post_lks.to_numpy(float), p.trust_post_acc.to_numpy(float))
    order = ['acc_above', 'equal', 'lks_above']
    trans = pd.crosstab(pd.Series(pre_rel, name='pre_relation'), pd.Series(post_rel, name='post_relation'))
    trans = trans.reindex(index=order, columns=order, fill_value=0)
    trans.to_csv(tables / 'scenario_family_rank_relation_transitions.csv')

    # ---------------- Part B: family features ----------------
    pid = p.participant_id.to_numpy()
    w['risk'] = w.risk.astype(float)
    assert (w.risk.dropna() > 0).all()
    mat = profile_matrix(w, pid)
    rng_f = np.random.default_rng(SEED_FOLDS)
    ids = pid.copy()
    rng_f.shuffle(ids)
    foldmap = {i: k for k, part in enumerate(np.array_split(ids, 5)) for i in part}
    folds = np.array([foldmap[i] for i in mat.index])
    exc = pd.DataFrame(index=mat.index, columns=BLOCKS, dtype=float)
    late = exc.copy()
    peakrows = []
    for b in BLOCKS:
        for k in range(5):
            train = mat.loc[folds != k, b]
            test = mat.loc[folds == k, b]
            peak = int(train.mean().idxmax())
            exc.loc[test.index, b] = test[peak] - test[1]
            late.loc[test.index, b] = test[NPOS[b]] - test[peak]
            peakrows.append((b, k, peak, len(test)))
    pk = pd.DataFrame(peakrows, columns=['scenario', 'fold', 'peak_position_from_training', 'n_test'])
    assert pk.equals(peaks_saved[pk.columns].astype(pk.dtypes.to_dict())), 'Fold peaks differ from saved'
    recon = pd.DataFrame({'participant_id': mat.index, 'mean_risk': mat.mean(axis=1).to_numpy(),
                          'excursion': exc.mean(axis=1).to_numpy(), 'late_change': late.mean(axis=1).to_numpy()})
    chk = recon.merge(saved, on='participant_id', suffixes=('_r', '_s'))
    for c in ['mean_risk', 'excursion', 'late_change']:
        assert np.allclose(chk[c + '_r'].to_numpy(), chk[c + '_s'].to_numpy(), atol=1e-9, equal_nan=True), c
    feats = pd.DataFrame({'participant_id': mat.index,
                          'mean_lat': mat[lat].mean(axis=1).to_numpy(), 'mean_long': mat[long_].mean(axis=1).to_numpy(),
                          'excursion_lat': exc[lat].mean(axis=1).to_numpy(), 'excursion_long': exc[long_].mean(axis=1).to_numpy(),
                          'late_lat': late[lat].mean(axis=1).to_numpy(), 'late_long': late[long_].mean(axis=1).to_numpy()})
    d = p.merge(saved[['participant_id'] + POOLED], on='participant_id', how='left', validate='one_to_one', sort=False)
    d = d.merge(feats, on='participant_id', how='left', validate='one_to_one', sort=False)
    assert np.array_equal(d.participant_id.to_numpy(), pid)
    d['D'] = (d.trust_post_lks - d.trust_pre_lks) - (d.trust_post_acc - d.trust_pre_acc)
    scaling = {}
    for c in RISK7 + POOLED[:3]:
        m, s = float(d[c].mean()), float(d[c].std(ddof=1))
        assert s > 0
        d[c + '_z'] = (d[c] - m) / s
        scaling[c] = {'mean': m, 'sd_ddof1': s, 'n_available': int(d[c].notna().sum())}
    for src, tgt in [('age', 'age_10'), ('driving_exp_years', 'experience_10')]:
        d[tgt] = (d[src] - float(d[src].mean())) / 10
    d['gender'] = d.gender.fillna('Missing').astype(str)
    d['country'] = d.country.fillna('Unknown').astype(str)
    req_pooled = ['D', 'trust_post_lks', 'trust_post_acc'] + PRIOR + [c + '_z' for c in POOLED] + ['age_10', 'experience_10']
    req = req_pooled + [c + '_z' for c in FAMILY]
    comp_pooled = d.dropna(subset=req_pooled).copy()
    comp = d.dropna(subset=req).copy()
    missing_family = {c: int(d[c].isna().sum()) for c in FAMILY}
    fam_corr = comp[[c + '_z' for c in FAMILY] + ['mean_risk_z', 'excursion_z', 'late_change_z']].corr()
    fam_corr.to_csv(tables / 'scenario_family_feature_correlations.csv', float_format='%.6g')

    # Pooled reproduction on the v5 rows, then refit on the family rows.
    f_pooled = pooled_plan['formula']
    y3, X3 = patsy.dmatrices(f_pooled, comp_pooled, return_type='dataframe', NA_action='raise')
    fit3 = hc1_fit(y3.iloc[:, 0], X3)
    rep_err = {}
    for t in ['mean_risk_z', 'excursion_z', 'late_change_z']:
        s = pooled_saved.set_index('term').loc[t]
        rep_err[t] = {'estimate_abs_err': float(abs(fit3.params[t] - s.estimate)), 'se_abs_err': float(abs(fit3.bse[t] - s.se_HC1))}
        assert max(rep_err[t].values()) < 1e-6, ('Pooled model not reproduced', t, rep_err[t])
    assert int(fit3.nobs) == int(pooled_saved.n.iloc[0]) == len(comp_pooled)
    y3b, X3b = patsy.dmatrices(f_pooled, comp, return_type='dataframe', NA_action='raise')
    fit3b = hc1_fit(y3b.iloc[:, 0], X3b)
    c3b = coef_rows(fit3b, X3b, 'D pooled model refit on family complete-case rows', ['mean_risk_z', 'excursion_z', 'late_change_z'])
    c3b.to_csv(tables / 'scenario_family_pooled_refit.csv', index=False, float_format='%.17g')

    # D model with family features.
    f1 = plan['formula_D_family']
    y1, X1 = patsy.dmatrices(f1, comp, return_type='dataframe', NA_action='raise')
    assert np.linalg.matrix_rank(X1.to_numpy(float)) == X1.shape[1]
    fit1 = hc1_fit(y1.iloc[:, 0], X1)
    famz = [c + '_z' for c in FAMILY]
    c1 = coef_rows(fit1, X1, 'D=(LKSpost-LKSpre)-(ACCpost-ACCpre)', famz)
    m1 = c1.role.eq('risk_family')
    c1['q_BH_family6'] = np.nan
    c1.loc[m1, 'q_BH_family6'] = multipletests(c1.loc[m1, 'p_two_sided_normal'], method='fdr_bh')[1]
    c1.to_csv(tables / 'scenario_family_D_coefficients.csv', index=False, float_format='%.17g')
    eq = pd.DataFrame([wald_equal(fit1, a + '_lat_z', a + '_long_z') for a in ['mean', 'excursion', 'late']])
    eq['q_BH_3'] = multipletests(eq.p_two_sided_normal, method='fdr_bh')[1]
    eq['model'] = 'M1_D_family'
    eq.to_csv(tables / 'scenario_family_D_equality_tests.csv', index=False, float_format='%.17g')

    # Own-baseline post-task models on the same rows and joint-HC1 contrasts.
    out2, fits2, designs2 = [], {}, {}
    for item in ['acc', 'lks']:
        f2 = plan['formula_post_item'].format(item=item)
        y2, X2 = patsy.dmatrices(f2, comp, return_type='dataframe', NA_action='raise')
        fits2[item] = hc1_fit(y2.iloc[:, 0], X2)
        designs2[item] = X2
        out2.append(coef_rows(fits2[item], X2, f'trust_post_{item}', famz))
    c2 = pd.concat(out2, ignore_index=True)
    m2 = c2.role.eq('risk_family')
    c2['q_BH_family12'] = np.nan
    c2.loc[m2, 'q_BH_family12'] = multipletests(c2.loc[m2, 'p_two_sided_normal'], method='fdr_bh')[1]
    c2.to_csv(tables / 'scenario_family_post_trust_coefficients.csv', index=False, float_format='%.17g')
    Xa, Xl = designs2['acc'].to_numpy(float), designs2['lks'].to_numpy(float)
    ra, rl = np.asarray(fits2['acc'].resid), np.asarray(fits2['lks'].resid)
    n_, ka = Xa.shape
    fac = n_ / (n_ - ka)
    Pa, Pl = np.linalg.pinv(Xa), np.linalg.pinv(Xl)
    cross = ((Pl * (rl * ra)[None, :]) @ Pa.T) * fac
    xrows = []
    for t in famz:
        ia, il = designs2['acc'].columns.get_loc(t), designs2['lks'].columns.get_loc(t)
        est = float(fits2['lks'].params[t] - fits2['acc'].params[t])
        var = float(fits2['lks'].cov_params().iloc[il, il] + fits2['acc'].cov_params().iloc[ia, ia] - 2 * cross[il, ia])
        z = est / np.sqrt(var)
        xrows.append({'term': t, 'lks_minus_acc_estimate': est, 'se_joint_HC1': float(np.sqrt(var)),
                      'ci_low': est - 1.959964 * np.sqrt(var), 'ci_high': est + 1.959964 * np.sqrt(var),
                      'p_two_sided_normal': float(2 * scipy.stats.norm.sf(abs(z))), 'n': n_})
    xc = pd.DataFrame(xrows)
    xc['q_BH_6'] = multipletests(xc.p_two_sided_normal, method='fdr_bh')[1]
    xc.to_csv(tables / 'scenario_family_post_trust_cross_outcome_contrasts.csv', index=False, float_format='%.17g')

    outputs = sorted(tables.glob('scenario_family_*.csv'))
    report = {'status': 'pass', 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'plan_sha256': sha(PLAN_PATH), 'pooled_plan_sha256': sha(POOLED_PLAN_PATH),
              'analysis_identity': plan['analysis_identity'],
              'n_complete_cases_family_models': int(fit1.nobs), 'n_excluded_complete_case': int(len(d) - fit1.nobs),
              'n_complete_cases_pooled_reproduction': int(fit3.nobs), 'missing_family_feature_counts': missing_family,
              'reconstruction_validation': {'fold_peaks_match_saved': True, 'pooled_features_match_saved_atol': 1e-9,
                                            'pooled_D_model_reproduction_abs_errors': rep_err},
              'r_squared_in_sample_descriptive_only': {'D_family': float(fit1.rsquared), 'D_pooled_v5_rows': float(fit3.rsquared),
                                                       'D_pooled_family_rows': float(fit3b.rsquared)},
              'formulas': {'D_family': f1, 'D_pooled': f_pooled, 'post_item': plan['formula_post_item']},
              'scaling': scaling, 'interpretation_limits': plan['interpretation_limits'],
              'source_sha256': source_sha, 'script_sha256': sha(Path(__file__)),
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                          'scipy': scipy.__version__, 'statsmodels': statsmodels.__version__, 'patsy': patsy.__version__},
              'outputs': {relative_label(x, root): {'sha256': sha(x), 'bytes': x.stat().st_size} for x in outputs}}
    for path in sources.values():
        assert sha(path) == source_sha[relative_label(path, root)], ('Source changed during analysis', path)
    dump(reports / 'scenario_family_report.json', report)
    pd.set_option('display.width', 200)
    print(direction.to_string(index=False))
    print(trans)
    print(c1.loc[m1].to_string(index=False))
    print(eq.to_string(index=False))
    print(xc.to_string(index=False))
    print(json.dumps({k: report[k] for k in ['n_complete_cases_family_models', 'n_complete_cases_pooled_reproduction',
                                             'r_squared_in_sample_descriptive_only']}, indent=2))


if __name__ == '__main__':
    main()
