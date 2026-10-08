#!/usr/bin/env python3
"""One frozen, post-review exploratory differential-trust-change association.

No peak selection, prediction search, model selection or original-data mutation.
The two post-outcome regressions below are algebra checks, not additional tests.
"""
from settings import WORKSPACE as ROOT, DATA_DIR, QUESTIONNAIRE, REPO, relative_label
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import patsy
import scipy
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


RISK = ['mean_risk', 'excursion', 'late_change', 'positive_fraction']
PRIOR = ['trust_pre_overall', 'trust_pre_acc', 'trust_pre_lks',
         'accept_pre_delegate', 'accept_pre_monitor_raw', 'accept_pre_distract']
PLAN_SHA = hashlib.sha256((REPO/'config/trust_contrast.json').read_bytes()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', type=Path, default=ROOT)
    args = ap.parse_args()
    root = args.project_root.resolve()
    out = root / 'outputs/parameters'
    reports, tables = out / 'reports', out / 'tables'
    reports.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    plan_path = REPO/'config/trust_contrast.json'
    plan = json.loads(plan_path.read_text())
    plan['source_sha256'] = {relative_label(p,root):sha(p) for p in [DATA_DIR/'participants.parquet',root/'outputs/core/tables/a2_participant_risk_features.csv']}
    p = pd.read_parquet(DATA_DIR / 'participants.parquet')
    f = pd.read_csv(root / 'outputs/core/tables/a2_participant_risk_features.csv')
    assert p.participant_id.is_unique and f.participant_id.is_unique
    assert set(p.participant_id) == set(f.participant_id)
    d = p.merge(f, on='participant_id', how='left', validate='one_to_one', sort=False)
    assert np.array_equal(d.participant_id, p.participant_id), 'Merge changed participant order'
    d['D'] = (d.trust_post_lks - d.trust_pre_lks) - (d.trust_post_acc - d.trust_pre_acc)

    scaling = {}
    for name in RISK:
        mean, sd = float(d[name].mean()), float(d[name].std(ddof=1))
        assert np.isfinite(mean) and np.isfinite(sd) and sd > 0
        d[name + '_z'] = (d[name] - mean) / sd
        scaling[name] = {'mean_all_available': mean, 'sd_all_available_ddof1': sd,
                         'n_available': int(d[name].notna().sum())}
    for source, target in [('age', 'age_10'), ('driving_exp_years', 'experience_10')]:
        mean = float(d[source].mean())
        d[target] = (d[source] - mean) / 10
        scaling[source] = {'mean_all_available': mean, 'divisor': 10}
    d['gender'] = d.gender.fillna('Missing').astype(str)
    d['country'] = d.country.fillna('Unknown').astype(str)
    required = ['D', 'trust_post_lks', 'trust_post_acc'] + PRIOR + [x + '_z' for x in RISK] + ['age_10', 'experience_10']
    assert not np.isinf(d[required].to_numpy(dtype=float)).any()
    missing = {x: int(d[x].isna().sum()) for x in required}
    full_description = {'scope': 'All participants with observed D, before model complete-case selection',
                        'n': int(d.D.notna().sum()), 'mean_D': float(d.D.mean()),
                        'sd_D_ddof1': float(d.D.std(ddof=1))}
    complete = d.dropna(subset=required).copy()
    y, design = patsy.dmatrices(plan['formula'], complete, return_type='dataframe', NA_action='raise')
    assert y.index.equals(complete.index) and design.index.equals(complete.index)
    X = design.to_numpy(dtype=float)
    Y = y.iloc[:, 0].to_numpy(dtype=float)
    n, k = X.shape
    rank = int(np.linalg.matrix_rank(X))
    assert rank == k, 'Rank-deficient design requires reporting rather than silent variable removal'
    fitted = sm.OLS(Y, design).fit(cov_type='HC1', use_t=False)
    ci = fitted.conf_int(alpha=.05)
    rows = []
    for term in design.columns:
        role = 'risk_primary_BH3' if term in [v + '_z' for v in RISK[:3]] else 'adjustment'
        rows.append({'term': term, 'estimate': float(fitted.params[term]),
                     'se_HC1': float(fitted.bse[term]), 'ci_low': float(ci.loc[term, 0]),
                     'ci_high': float(ci.loc[term, 1]), 'p_two_sided_normal': float(fitted.pvalues[term]),
                     'q_BH_risk3': np.nan, 'role': role, 'n': n,
                     'outcome': '(LKSpost-LKSpre)-(ACCpost-ACCpre)'})
    coefficients = pd.DataFrame(rows)
    mask = coefficients.role.eq('risk_primary_BH3')
    assert mask.sum() == 3
    coefficients.loc[mask, 'q_BH_risk3'] = multipletests(
        coefficients.loc[mask, 'p_two_sided_normal'], method='fdr_bh')[1]

    # Independent ordinary least-squares solve and HC1 sandwich.
    beta_numpy = np.linalg.lstsq(X, Y, rcond=None)[0]
    residual_numpy = Y - X @ beta_numpy
    pinv_x = np.linalg.pinv(X)
    hc1_factor = n / (n - rank)
    cov_numpy = ((pinv_x * residual_numpy[None, :]) @
                 (pinv_x * residual_numpy[None, :]).T) * hc1_factor

    # Algebra validation uses EXACTLY the same rows and design matrix.
    lks = sm.OLS(complete.trust_post_lks.to_numpy(), design).fit(cov_type='HC1', use_t=False)
    acc = sm.OLS(complete.trust_post_acc.to_numpy(), design).fit(cov_type='HC1', use_t=False)
    correction = np.zeros(k)
    correction[design.columns.get_loc('trust_pre_lks')] = -1
    correction[design.columns.get_loc('trust_pre_acc')] = 1
    expected_beta = lks.params.to_numpy() - acc.params.to_numpy() + correction
    el, ea = np.asarray(lks.resid), np.asarray(acc.resid)
    cross_cov = ((pinv_x * (el * ea)[None, :]) @ pinv_x.T) * hc1_factor
    expected_cov = lks.cov_params().to_numpy() + acc.cov_params().to_numpy() - cross_cov - cross_cov.T
    cov_d = fitted.cov_params().to_numpy()
    errors = {
        'numpy_lstsq_vs_statsmodels_coefficients': float(np.max(np.abs(beta_numpy - fitted.params))),
        'independent_HC1_vs_statsmodels_covariance': float(np.max(np.abs(cov_numpy - cov_d))),
        'D_vs_post_difference_corrected_all_coefficients': float(np.max(np.abs(expected_beta - fitted.params))),
        'D_vs_post_difference_residuals': float(np.max(np.abs(fitted.resid - (el - ea)))),
        'D_HC1_vs_joint_post_difference_HC1_covariance': float(np.max(np.abs(expected_cov - cov_d)))
    }
    assert max(errors.values()) < 1e-8, errors
    algebra = []
    for idx, term in enumerate(design.columns):
        algebra.append({'term': term, 'D_estimate': float(fitted.params.iloc[idx]),
                        'post_lks_estimate': float(lks.params.iloc[idx]),
                        'post_acc_estimate': float(acc.params.iloc[idx]),
                        'known_baseline_subtraction_coefficient': float(correction[idx]),
                        'post_lks_minus_acc_plus_baseline_subtraction': float(expected_beta[idx]),
                        'coefficient_absolute_error': float(abs(expected_beta[idx] - fitted.params.iloc[idx])),
                        'D_HC1_variance': float(cov_d[idx, idx]),
                        'joint_post_difference_HC1_variance': float(expected_cov[idx, idx])})
    filenames = []
    for suffix, frame in [('coefficients', coefficients), ('risk3', coefficients.loc[mask].copy()),
                          ('algebra_check', pd.DataFrame(algebra))]:
        path = tables / f'within_person_trust_contrast_{suffix}.csv'
        frame.to_csv(path, index=False, float_format='%.17g')
        filenames.append(path)
    scalepath = reports / 'within_person_trust_contrast_scaling.json'
    dump(scalepath, scaling)
    filenames.append(scalepath)
    validpath = reports / 'within_person_trust_contrast_validation.json'
    validation = {'status': 'pass', 'n_identical_design_rows': n, 'design_rank': rank,
                  'design_columns': k, 'tolerance': 1e-8, 'maximum_absolute_errors': errors,
                  'risk_coefficients_equal_post_lks_minus_post_acc': True,
                  'all_coefficients_identity_includes_explicit_pre_item_subtraction': True,
                  'joint_covariance_formula': 'Cov(LKS)+Cov(ACC)-Cov(LKS,ACC)-Cov(ACC,LKS); HC1 n/(n-rank(X)); same X and rows',
                  'plan_sha256': PLAN_SHA}
    dump(validpath, validation)
    filenames.append(validpath)
    report = {'status': 'pass', 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'analysis_identity': plan['analysis_identity'], 'formula': plan['formula'],
              'n_source_participants': len(d), 'n_complete_cases': n, 'n_excluded_complete_case': len(d) - n,
              'complete_case_missing_counts_by_variable': missing,
              'D_complete_case_mean': float(complete.D.mean()),
              'D_complete_case_SD_ddof1': float(complete.D.std(ddof=1)),
              'D_all_observed_descriptive': full_description,
              'r_squared_in_sample': float(fitted.rsquared),
              'adjusted_r_squared_in_sample': float(fitted.rsquared_adj),
              'performance_label': 'In-sample descriptive fit of one associational OLS model; NOT held-out prediction',
              'design_rank': rank, 'df_resid': float(fitted.df_resid),
              'condition_number': float(fitted.condition_number),
              'covariance': 'HC1, normal 95% intervals, two-sided normal tests',
              'BH_family': [x + '_z' for x in RISK[:3]], 'BH_family_size': 3,
              'numeric_predictor_scaling': scaling,
              'gender_levels': sorted(complete.gender.unique()), 'country_levels': sorted(complete.country.unique()),
              'complete_case_participant_ids_sha256': hashlib.sha256(json.dumps(complete.participant_id.tolist(), separators=(',', ':')).encode()).hexdigest(),
              'design_matrix_sha256': hashlib.sha256(np.ascontiguousarray(X, dtype='<f8').tobytes()).hexdigest(),
              'design_columns': list(design.columns), 'no_peak_reselection': True,
              'no_variable_or_threshold_search': True, 'no_prediction_search': True,
              'scientific_model_count': 1, 'post_outcome_models_are_algebra_checks_only': True,
              'source_sha256': plan['source_sha256'], 'plan_sha256': PLAN_SHA,
              'script_sha256': sha(Path(__file__)), 'validation': validation,
              'interpretation_limits': plan['interpretation_limits'],
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                          'scipy': scipy.__version__, 'statsmodels': statsmodels.__version__, 'patsy': patsy.__version__},
              'outputs': {str(path.relative_to(root)): {'sha256': sha(path), 'bytes': path.stat().st_size} for path in filenames}}
    for rel, digest in plan['source_sha256'].items():
        assert sha(Path(rel) if Path(rel).is_absolute() else root / rel) == digest, ('Source changed during analysis', rel)
    dump(reports / 'within_person_trust_contrast_report.json', report)
    print(coefficients.loc[mask].to_string(index=False))
    print(json.dumps({key: report[key] for key in ['n_complete_cases', 'D_complete_case_mean',
                         'D_complete_case_SD_ddof1', 'r_squared_in_sample']}, indent=2))
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
