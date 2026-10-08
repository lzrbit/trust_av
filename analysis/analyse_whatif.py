#!/usr/bin/env python3
"""Frozen-plan held-out model what-if sensitivity. Never trains a model.

Run with Python 3.12 and the round-2 model requirements. Existing local joblib
bundles must be locally trusted. Current input hashes and held-out prediction
identity are verified before perturbation; this is not preregistration.
"""
from __future__ import annotations
from settings import WORKSPACE as ROOT, DATA_DIR, QUESTIONNAIRE, REPO, relative_label
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

NUMERIC = ['age', 'driving_exp_years', 'trust_pre_overall', 'trust_pre_acc', 'trust_pre_lks',
           'accept_pre_delegate', 'accept_pre_monitor_raw', 'accept_pre_distract']
ITEMS = NUMERIC[2:]
CATEGORICAL = ['gender', 'education', 'country', 'own_car', 'drive_freq_12m', 'km_life', 'km_12m',
               'drive_type', 'drive_style', 'know_ad', 'acc_years', 'acc_dist', 'lks_years', 'lks_dist',
               'acclks_years', 'acclks_dist']
IDENTITY = ['scenario', 'event_key']
ACC, LKS = 'trust_pre_acc', 'trust_pre_lks'
DISPLAY = dict(zip(ITEMS, ['Initial overall trust', 'Initial ACC trust', 'Initial LKS trust',
                         'Initial delegation willingness', 'Initial need to monitor',
                         'Initial distraction willingness']))
K = 100
COUNTS = (5, 10, 20)
MIN_N = 30
B = 2000
SEED = 20261007


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(obj, path):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def read_data(root):
    base = root / 'outputs/core'
    p = pd.read_parquet(DATA_DIR / 'participants.parquet')
    e = pd.read_parquet(DATA_DIR / 'events.parquet')
    assert p.participant_id.is_unique and e.event_row_id.is_unique
    p = p[['participant_id'] + NUMERIC + CATEGORICAL].copy()
    for c in NUMERIC:
        p[c] = pd.to_numeric(p[c], errors='coerce').astype(float)
    for c in CATEGORICAL:
        p[c] = p[c].fillna('__missing__').astype(str)
    p = p.sort_values('participant_id').reset_index(drop=True)
    # All six original initial-state scores must be observed integer values.
    assert np.isfinite(p[ITEMS]).all().all()
    assert ((p[ITEMS] >= 0) & (p[ITEMS] <= 10)).all().all()
    assert np.equal(p[ITEMS], np.round(p[ITEMS])).all().all()
    e = e[e.risk_event_mean.notna()].copy()
    e = e[['event_row_id', 'participant_id', 'scenario', 'event_key', 'risk_event_mean']]
    data = e.rename(columns={'event_row_id': 'row_key'}).merge(p, on='participant_id', validate='m:1')
    for c in IDENTITY:
        data[c] = data[c].fillna('__missing__').astype(str)
    data = data.sort_values(['participant_id', 'scenario', 'event_key']).reset_index(drop=True)
    assert data.row_key.is_unique and data.participant_id.nunique() == len(p)
    oof = pd.read_parquet(base / 'prediction/m2_oof_predictions.parquet')
    oof = oof[(oof.protocol == 'participant_5fold') & (oof.model == 'LightGBM')].copy()
    oof['fold'] = oof.fold.astype(str)
    assert oof.row_key.is_unique and set(oof.row_key) == set(data.row_key)
    membership = pd.read_parquet(base / 'prediction/m2_fold_membership.parquet')
    merged = oof[['row_key', 'participant_id', 'fold']].merge(
        membership[['row_key', 'participant_id', 'participant_fold']], on=['row_key', 'participant_id'], validate='1:1')
    assert len(merged) == len(data)
    assert (merged.fold == merged.participant_fold.astype(str)).all()
    assert merged.groupby('participant_id').fold.nunique().eq(1).all()
    p = p.merge(merged[['participant_id', 'fold']].drop_duplicates(), on='participant_id', validate='1:1')
    return p, data, oof


def numeric_contribution(a, b, scale):
    diff = np.abs(a[:, None] - b[None, :])
    return np.minimum(diff / scale, 1.) if scale > 0 else (diff > 0).astype(float)


def support_fold(train, test, fold):
    """Unique-person support; no outcomes, predictions or test-fitted parameters."""
    assert train.participant_id.is_unique and test.participant_id.is_unique
    assert not set(train.participant_id) & set(test.participant_id)
    train = train.sort_values('participant_id').reset_index(drop=True)
    test = test.sort_values('participant_id').reset_index(drop=True)
    nt, nq = len(train), len(test)
    total_tt = np.zeros((nt, nt)); total_qt = np.zeros((nq, nt))
    numeric = {}
    params = []
    for c in NUMERIC:
        tr = train[c].to_numpy(copy=True); q = test[c].to_numpy(copy=True)
        median = float(np.nanmedian(tr)) if np.isfinite(tr).any() else 0.
        missing_tr, missing_q = ~np.isfinite(tr), ~np.isfinite(q)
        tr[missing_tr] = median; q[missing_q] = median
        scale = float(np.max(tr) - np.min(tr))
        numeric[c] = (tr, q, scale)
        total_tt += numeric_contribution(tr, tr, scale)
        total_qt += numeric_contribution(q, tr, scale)
        params.append(dict(fold=fold, feature=c, median=median, minimum=float(np.min(tr)), maximum=float(np.max(tr)),
                           range=scale, train_imputed=int(missing_tr.sum()), test_imputed=int(missing_q.sum())))
    for c in CATEGORICAL:
        tr = train[c].to_numpy(); q = test[c].to_numpy()
        total_tt += (tr[:, None] != tr[None, :])
        total_qt += (q[:, None] != tr[None, :])
    all_support, all_calibration, all_neighbors = [], [], []
    for context, excluded in [(x, [x]) for x in ITEMS] + [('joint_acc_lks', [ACC, LKS])]:
        tt, qt = total_tt.copy(), total_qt.copy()
        for c in excluded:
            tr, q, scale = numeric[c]
            tt -= numeric_contribution(tr, tr, scale)
            qt -= numeric_contribution(q, tr, scale)
        # Round-off can produce tiny negative values after subtracting terms.
        tt = np.maximum(tt, 0.) / (24 - len(excluded))
        qt = np.maximum(qt, 0.) / (24 - len(excluded))
        np.fill_diagonal(tt, np.inf)
        loo_d100 = np.partition(tt, K - 1, axis=1)[:, K - 1]
        radius = float(np.quantile(loo_d100, .95, method='linear'))
        nn = np.argsort(qt, axis=1, kind='stable')[:, :K]
        distances = np.take_along_axis(qt, nn, axis=1)
        d100 = distances[:, -1]
        support = test[['participant_id', 'country', 'fold']].copy()
        support['context'] = context
        support['d100'] = d100
        support['training_loo_d100_p95'] = radius
        support['radius_supported'] = d100 <= radius
        support['n_missing_numeric_distance'] = test[[c for c in NUMERIC if c not in excluded]].isna().sum(axis=1).to_numpy()
        if context != 'joint_acc_lks':
            tr_values = train[context].to_numpy()
            raw = test[context].to_numpy()
            support['original_value'] = raw
            for v in range(11):
                support[f'neighbor_count_{v}'] = (tr_values[nn] == v).sum(axis=1)
            counts = np.column_stack([support[f'neighbor_count_{v}'] for v in range(11)])
            for step in (-2, -1, 0, 1, 2):
                values = raw + step
                boundary = (values >= 0) & (values <= 10)
                # Invalid values get count 0 rather than indexing outside the range.
                n = counts[np.arange(nq), np.clip(values, 0, 10).astype(int)]
                n[~boundary] = 0
                support[f'count_step_{step}'] = n
                support[f'boundary_step_{step}'] = boundary
            for min_count in COUNTS:
                support[f'primary_eligible_m{min_count}'] = (
                    support.radius_supported & (raw >= 1) & (raw <= 9) &
                    (support[['count_step_-1', 'count_step_0', 'count_step_1']].min(axis=1) >= min_count))
            support['dose_eligible_m10'] = (
                support.radius_supported & (raw >= 2) & (raw <= 8) &
                (support[[f'count_step_{s}' for s in (-2, -1, 0, 1, 2)]].min(axis=1) >= 10))
        else:
            a, l = test[ACC].to_numpy(), test[LKS].to_numpy()
            train_a, train_l = train[ACC].to_numpy()[nn], train[LKS].to_numpy()[nn]
            support['original_acc'], support['original_lks'] = a, l
            for name, da, dl in [('base', 0, 0), ('acc', 1, 0), ('lks', 0, 1), ('both', 1, 1)]:
                support[f'corner_count_{name}'] = ((train_a == (a + da)[:, None]) & (train_l == (l + dl)[:, None])).sum(axis=1)
            support['joint_boundary'] = (a <= 9) & (l <= 9)
            support['joint_eligible_m10'] = (support.radius_supported & support.joint_boundary &
                (support[[f'corner_count_{v}' for v in ['base', 'acc', 'lks', 'both']]].min(axis=1) >= 10))
        all_support.append(support)
        all_calibration.append(pd.DataFrame(dict(participant_id=train.participant_id.to_numpy(), fold=fold,
                                                context=context, loo_d100=loo_d100, training_loo_d100_p95=radius)))
        all_neighbors.append(pd.DataFrame(dict(participant_id=np.repeat(test.participant_id.to_numpy(), K), fold=fold,
            context=context, rank=np.tile(np.arange(1, K + 1), nq),
            training_neighbor_id=train.participant_id.to_numpy()[nn].ravel(), distance=distances.ravel())))
    return pd.concat(all_support, ignore_index=True), pd.concat(all_calibration, ignore_index=True), pd.concat(all_neighbors, ignore_index=True), params


def baseline_validation(root, data, oof, reports):
    base = root / 'outputs/core/prediction/models'
    lookup = data.set_index('row_key', drop=False)
    bundles, rows, checks = {}, [], []
    for fold in range(5):
        bundle = joblib.load(base / f'm2_participant_fold_{fold}.joblib')
        assert bundle['numeric'] == NUMERIC and bundle['categorical'] == CATEGORICAL + IDENTITY
        train = lookup.loc[bundle['train_row_keys']]
        test = lookup.loc[bundle['test_row_keys']]
        assert len(set(bundle['train_row_keys'])) == len(train)
        assert len(set(bundle['test_row_keys'])) == len(test)
        assert not set(train.row_key) & set(test.row_key)
        assert not set(train.participant_id) & set(test.participant_id)
        assert set(train.row_key) | set(test.row_key) == set(data.row_key)
        ref = oof[oof.fold == str(fold)].set_index('row_key')
        assert set(test.row_key) == set(ref.index)
        assert (test.participant_id.to_numpy() == ref.loc[test.row_key].participant_id.to_numpy()).all()
        assert np.allclose(test.risk_event_mean, ref.loc[test.row_key, 'actual'], rtol=0, atol=1e-12)
        pred = bundle['pipeline'].predict(test)
        error = pred - ref.loc[test.row_key, 'pred'].to_numpy()
        assert np.isfinite(pred).all() and np.max(np.abs(error)) <= 1e-10
        row = test[['row_key', 'participant_id', 'scenario', 'event_key']].copy().reset_index(drop=True)
        row['fold'] = str(fold); row['pred_original'] = pred
        row['saved_oof_pred'] = ref.loc[test.row_key, 'pred'].to_numpy()
        row['oof_abs_error'] = np.abs(error)
        rows.append(row)
        checks.append(dict(fold=fold, n_train_people=int(train.participant_id.nunique()), n_test_people=int(test.participant_id.nunique()),
                           n_train_rows=len(train), n_test_rows=len(test), max_absolute_oof_error=float(np.max(np.abs(error))),
                           participant_overlap=0, row_key_sets_match=True))
        bundles[fold] = bundle
    original = pd.concat(rows, ignore_index=True)
    assert original.row_key.is_unique and len(original) == len(data)
    dump(dict(status='passed', checks=checks, n_rows=len(original), n_participants=int(original.participant_id.nunique()),
              max_absolute_oof_error=float(original.oof_abs_error.max())), reports / 'whatif_oof_reproduction.json')
    return bundles, original


def generate_predictions(data, bundles, baseline, support, destination):
    settings = [('baseline', '', 0, '', 0)]
    settings += [(f'{item}__{step:+d}', item, step, '', 0) for item in ITEMS for step in (-1, 1)]
    settings += [(f'{item}__{step:+d}', item, step, '', 0) for item in (ACC, LKS) for step in (-2, 2)]
    settings += [('joint_acc_lks__+1_+1', ACC, 1, LKS, 1)]
    lookup = data.set_index('row_key', drop=False)
    base_lookup = baseline.set_index('row_key')
    support_lookup = support.set_index(['participant_id', 'context'])
    rows = []
    for fold, bundle in bundles.items():
        test = lookup.loc[bundle['test_row_keys']].copy().reset_index(drop=True)
        for setting, item1, step1, item2, step2 in settings:
            changed = test.copy()
            rec = test[['row_key', 'participant_id', 'scenario', 'event_key', 'country']].copy()
            rec['fold'] = str(fold); rec['setting'] = setting
            rec['item1'] = item1; rec['step1'] = step1; rec['item2'] = item2; rec['step2'] = step2
            rec['original_value1'] = test[item1].to_numpy() if item1 else np.nan
            rec['changed_value1'] = rec.original_value1 + step1
            rec['original_value2'] = test[item2].to_numpy() if item2 else np.nan
            rec['changed_value2'] = rec.original_value2 + step2
            valid = np.ones(len(test), dtype=bool)
            for item, step in [(item1, step1), (item2, step2)]:
                if item:
                    changed[item] = test[item] + step
                    valid &= changed[item].between(0, 10).to_numpy()
            rec['boundary_valid'] = valid
            pred_base = base_lookup.loc[test.row_key, 'pred_original'].to_numpy()
            pred = np.full(len(test), np.nan)
            if item1:
                pred[valid] = bundle['pipeline'].predict(changed.loc[valid])
            else:
                pred = pred_base.copy()
            assert np.isfinite(pred[valid]).all()
            rec['pred_original'], rec['pred_changed'] = pred_base, pred
            rec['prediction_delta'] = pred - pred_base
            if item1:
                context = 'joint_acc_lks' if item2 else item1
                sup = support_lookup.loc[pd.MultiIndex.from_arrays([test.participant_id, [context] * len(test)])]
                for c in ['d100', 'training_loo_d100_p95', 'radius_supported', 'primary_eligible_m5',
                          'primary_eligible_m10', 'primary_eligible_m20', 'dose_eligible_m10', 'joint_eligible_m10']:
                    rec[c] = sup[c].to_numpy()
                rec['context'] = context
                if item2:
                    rec['neighbor_count_original'] = sup.corner_count_base.to_numpy()
                    rec['neighbor_count_changed'] = sup.corner_count_both.to_numpy()
                else:
                    rec['neighbor_count_original'] = sup['count_step_0'].to_numpy()
                    rec['neighbor_count_changed'] = sup[f'count_step_{step1}'].to_numpy()
            else:
                rec['context'] = 'baseline'
            rows.append(rec)
        print(f'PREDICT fold={fold} rows={len(test)} settings={len(settings)}', flush=True)
    events = pd.concat(rows, ignore_index=True)
    assert not events.duplicated(['row_key', 'setting']).any()
    assert len(events) == len(data) * len(settings)
    events.to_parquet(destination / 'whatif_event_predictions.parquet', index=False)
    # Average each participant's observed events before any population averaging.
    agg = dict(pred_original=('pred_original', 'mean'), pred_changed=('pred_changed', 'mean'),
               prediction_delta=('prediction_delta', 'mean'), n_observed_events=('row_key', 'size'),
               n_predicted_events=('pred_changed', 'count'))
    keys = ['participant_id', 'fold', 'country', 'setting', 'context', 'item1', 'step1', 'item2', 'step2']
    person = events.groupby(keys, dropna=False, sort=False).agg(**agg).reset_index().assign(scope='pooled')
    blocks = events.groupby(keys + ['scenario'], dropna=False, sort=False).agg(**agg).reset_index().rename(columns={'scenario': 'scope'})
    person = pd.concat([person, blocks], ignore_index=True)
    extra = ['original_value1', 'changed_value1', 'original_value2', 'changed_value2', 'boundary_valid',
             'd100', 'training_loo_d100_p95', 'radius_supported', 'primary_eligible_m5', 'primary_eligible_m10',
             'primary_eligible_m20', 'dose_eligible_m10', 'joint_eligible_m10', 'neighbor_count_original', 'neighbor_count_changed']
    metadata = events[['participant_id', 'setting'] + extra].drop_duplicates()
    assert not metadata.duplicated(['participant_id', 'setting']).any()
    person = person.merge(metadata, on=['participant_id', 'setting'], validate='m:1')
    assert not person.duplicated(['participant_id', 'setting', 'scope']).any()
    assert (person.n_predicted_events.eq(person.n_observed_events) | person.n_predicted_events.eq(0)).all()
    person.to_parquet(destination / 'whatif_participant_predictions.parquet', index=False)
    return person, len(events)


class SharedBootstrap:
    def __init__(self, pids, destination):
        self.pids = pd.Index(pids, name='participant_id')
        self.weights = np.random.default_rng(SEED).multinomial(len(pids), np.full(len(pids), 1 / len(pids)), size=B)
        assert np.all(self.weights.sum(axis=1) == len(pids))
        np.savez_compressed(destination / 'whatif_shared_bootstrap_weights.npz', weights=self.weights)
        pd.DataFrame({'column': np.arange(len(pids)), 'participant_id': pids}).to_parquet(destination / 'whatif_bootstrap_participants.parquet', index=False)

    def summarize(self, values):
        aligned = values.reindex(self.pids).to_numpy(dtype=float)
        ok = np.isfinite(aligned)
        n = int(ok.sum())
        out = dict(n_eligible=n, estimate=np.nan, ci_low=np.nan, ci_high=np.nan, bootstrap_se=np.nan,
                   n_bootstrap_valid=0, status='insufficient_support' if n < MIN_N else 'estimated')
        if n < MIN_N:
            return out, None
        weights = self.weights[:, ok]
        den = weights.sum(axis=1)
        rep = np.divide(weights @ aligned[ok], den, out=np.full(B, np.nan), where=den > 0)
        valid = np.isfinite(rep)
        out.update(estimate=float(aligned[ok].mean()), ci_low=float(np.quantile(rep[valid], .025)),
                   ci_high=float(np.quantile(rep[valid], .975)), bootstrap_se=float(np.std(rep[valid], ddof=1)),
                   n_bootstrap_valid=int(valid.sum()))
        return out, rep


def summarize_results(p, support, person, tables, reports, destination):
    boot = SharedBootstrap(p.participant_id.to_numpy(copy=True), destination)
    support_idx = support.set_index(['context', 'participant_id']).sort_index()
    scopes = ['pooled'] + sorted(x for x in person.scope.unique() if x != 'pooled')
    person_idx = person.set_index(['setting', 'scope', 'participant_id']).sort_index()
    contrasts, primary_reps, primary_positions = [], [], []
    values_out = []

    def estimate(group, context, setting, scope, min_count, mask, key, value_override=None):
        if value_override is None:
            vals = person_idx.loc[(setting, scope), 'prediction_delta'].copy()
        else:
            vals = value_override.copy()
        eligible = mask.reindex(vals.index, fill_value=False).astype(bool)
        vals = vals.where(eligible)
        summary, rep = boot.summarize(vals)
        row = dict(analysis=group, context=context, setting=setting, scope=scope, min_count=min_count, contrast=key, **summary)
        if context in DISPLAY:
            row['item_label'] = DISPLAY[context]
        idx = len(contrasts); contrasts.append(row)
        if group == 'primary' and scope == 'pooled' and min_count == 10:
            primary_positions.append(idx); primary_reps.append(rep)
        if scope == 'pooled':
            values_out.append(pd.DataFrame({'participant_id': vals.index, 'analysis': group, 'context': context,
                                           'contrast': key, 'min_count': min_count, 'eligible': eligible, 'participant_delta': vals}))

    coverage = []
    for item in ITEMS:
        sub = support_idx.loc[item]
        raw = sub.original_value
        for m in COUNTS:
            elig = sub[f'primary_eligible_m{m}'].astype(bool)
            local_ok = sub[['count_step_-1', 'count_step_0', 'count_step_1']].min(axis=1) >= m
            boundary_ok = raw.between(1, 9)
            for country in ['ALL'] + sorted(sub.country.unique()):
                ss = np.ones(len(sub), dtype=bool) if country == 'ALL' else sub.country.eq(country).to_numpy()
                coverage.append(dict(analysis='primary', context=item, min_count=m, country=country,
                    n_total=int(ss.sum()), n_boundary_valid=int((boundary_ok & ss).sum()),
                    n_radius_valid=int((sub.radius_supported & ss).sum()), n_local_count_valid=int((local_ok & ss).sum()),
                    n_eligible=int((elig & ss).sum()), n_boundary_excluded=int((~boundary_ok & ss).sum()),
                    n_radius_excluded_after_boundary=int((boundary_ok & ~sub.radius_supported & ss).sum()),
                    n_count_excluded_after_boundary_radius=int((boundary_ok & sub.radius_supported & ~local_ok & ss).sum())))
            for step in (-1, 1):
                for scope in scopes:
                    estimate('primary' if m == 10 else 'support_sensitivity', item, f'{item}__{step:+d}', scope, m, elig, f'{step:+d}')
    # Simultaneous coverage belongs only to the prespecified 12 primary pooled contrasts.
    standardized = []; included = []; zero_se = []
    for idx, rep in zip(primary_positions, primary_reps):
        if rep is None:
            continue
        estimate_value, se = contrasts[idx]['estimate'], contrasts[idx]['bootstrap_se']
        assert np.isfinite(rep).all(), 'Missing shared-bootstrap replicate in a primary estimable contrast'
        if se <= 1e-14:
            assert np.max(np.abs(rep - estimate_value)) <= 1e-12, 'Zero SE with nonconstant replicates'
            zero_se.append(idx)
        else:
            standardized.append(np.abs((rep - estimate_value) / se)); included.append(idx)
    critical = float(np.quantile(np.max(np.vstack(standardized), axis=0), .95)) if standardized else 0.
    n_estimable = len(included) + len(zero_se)
    for idx in primary_positions:
        row = contrasts[idx]
        row['simultaneous_family_planned'] = 12
        row['simultaneous_family_estimable'] = n_estimable
        row['simultaneous_critical'] = critical
        row['simultaneous_family_complete'] = n_estimable == 12
        row['simultaneous_ci_low'] = row['estimate'] - critical * row['bootstrap_se']
        row['simultaneous_ci_high'] = row['estimate'] + critical * row['bootstrap_se']
        if idx in zero_se:
            row['simultaneous_ci_low'] = row['simultaneous_ci_high'] = row['estimate']
    for item in (ACC, LKS):
        sub = support_idx.loc[item]
        elig = sub.dose_eligible_m10.astype(bool)
        coverage.append(dict(analysis='dose', context=item, min_count=10, country='ALL', n_total=len(sub), n_eligible=int(elig.sum())))
        for step in (-2, -1, 0, 1, 2):
            setting = 'baseline' if step == 0 else f'{item}__{step:+d}'
            for scope in scopes:
                estimate('dose', item, setting, scope, 10, elig, f'{step:+d}')
    joint = support_idx.loc['joint_acc_lks']
    elig = joint.joint_eligible_m10.astype(bool)
    coverage.append(dict(analysis='joint', context='joint_acc_lks', min_count=10, country='ALL', n_total=len(joint), n_eligible=int(elig.sum())))
    for scope in scopes:
        a = person_idx.loc[(f'{ACC}__+1', scope), 'prediction_delta']
        l = person_idx.loc[(f'{LKS}__+1', scope), 'prediction_delta']
        both = person_idx.loc[('joint_acc_lks__+1_+1', scope), 'prediction_delta']
        for key, vals in [('joint_total', both), ('acc_only', a), ('lks_only', l), ('finite_interaction', both - a - l)]:
            estimate('joint', 'joint_acc_lks', 'joint_acc_lks__+1_+1', scope, 10, elig, key, vals)
    # Frozen pre-result addendum: describe within-primary-set response heterogeneity.
    strata = []
    for item in (ACC, LKS):
        sub = support_idx.loc[item]
        common = sub.primary_eligible_m10.astype(bool)
        for step in (-1, 1):
            values = person_idx.loc[(f'{item}__{step:+d}', 'pooled'), 'prediction_delta']
            for initial in range(11):
                mask = common & sub.original_value.eq(initial)
                vals = values.where(mask.reindex(values.index, fill_value=False))
                finite = vals.dropna().to_numpy()
                summary, _ = boot.summarize(vals)
                row = dict(context=item, item_label=DISPLAY[item], step=step, baseline_value=initial,
                           n_eligible=len(finite), mean=float(np.mean(finite)) if len(finite) else np.nan,
                           minimum=float(np.min(finite)) if len(finite) else np.nan,
                           maximum=float(np.max(finite)) if len(finite) else np.nan,
                           ci_low=summary['ci_low'], ci_high=summary['ci_high'],
                           ci_status=summary['status'], n_bootstrap_valid=summary['n_bootstrap_valid'])
                for quantile in [10, 25, 50, 75, 90]:
                    row[f'p{quantile}'] = float(np.quantile(finite, quantile / 100)) if len(finite) else np.nan
                strata.append(row)
    pd.DataFrame(strata).to_csv(tables / 'whatif_baseline_strata.csv', index=False)
    # Authorized post-primary exploration: compare features within identical people.
    common_acclks = (support_idx.loc[ACC, 'primary_eligible_m10'].astype(bool) &
                     support_idx.loc[LKS, 'primary_eligible_m10'].astype(bool))
    paired_rows = []
    paired_cache = []
    for step in (-1, 1):
        for scope in scopes:
            a = person_idx.loc[(f'{ACC}__{step:+d}', scope), 'prediction_delta']
            l = person_idx.loc[(f'{LKS}__{step:+d}', scope), 'prediction_delta']
            for label, vals in [('ACC', a), ('LKS', l), ('LKS_minus_ACC', l - a)]:
                vals = vals.where(common_acclks.reindex(vals.index, fill_value=False))
                summary, _ = boot.summarize(vals)
                paired_rows.append(dict(analysis='post_primary_exploratory', feature_contrast=label,
                    step=step, scope=scope, sd=float(vals.std(ddof=1)) if vals.notna().sum() >= MIN_N else np.nan, **summary))
                paired_cache.append(pd.DataFrame(dict(participant_id=vals.index, feature_contrast=label,
                    step=step, scope=scope, participant_delta=vals,
                    eligible=common_acclks.reindex(vals.index, fill_value=False))))
    pd.DataFrame(paired_rows).to_csv(tables / 'whatif_paired_feature.csv', index=False)
    pd.concat(paired_cache, ignore_index=True).to_parquet(destination / 'whatif_paired_feature_participants.parquet', index=False)
    all_results = pd.DataFrame(contrasts)
    all_results.to_csv(tables / 'whatif_all_contrasts.csv', index=False)
    for name, selector in [('primary', all_results.analysis.eq('primary')),
                           ('support_sensitivity', all_results.analysis.eq('support_sensitivity')),
                           ('dose', all_results.analysis.eq('dose')), ('joint', all_results.analysis.eq('joint'))]:
        all_results[selector].to_csv(tables / f'whatif_{name}.csv', index=False)
    pd.DataFrame(coverage).to_csv(tables / 'whatif_support_coverage.csv', index=False)
    pd.concat(values_out, ignore_index=True).to_parquet(destination / 'whatif_participant_contrasts.parquet', index=False)
    reps_table = pd.DataFrame({'bootstrap_draw': np.arange(B)})
    for idx, rep in zip(primary_positions, primary_reps):
        row = contrasts[idx]
        reps_table[f"{row['context']}__{row['contrast']}"] = rep if rep is not None else np.nan
    reps_table.to_parquet(destination / 'whatif_primary_bootstrap_replicates.parquet', index=False)
    dump(dict(n_planned=12, n_estimable=n_estimable, n_zero_se=len(zero_se), critical_value=critical,
              n_draws=B, seed=SEED, complete=n_estimable == 12,
              method='Fixed-SE standardized maximum absolute bootstrap deviation, pooled primary contrasts only'),
         reports / 'whatif_simultaneous_intervals.json')
    return all_results


def validate_outputs(p, support, person, dest, tables, reports):
    """Independent cache-level algebra and participant-bootstrap reconstruction."""
    events = pd.read_parquet(dest / 'whatif_event_predictions.parquet')
    neighbors = pd.read_parquet(dest / 'whatif_training_neighbors.parquet')
    original = pd.read_parquet(dest / 'whatif_baseline_oof_reproduction.parquet').set_index('row_key')
    source = p.set_index('participant_id')
    assert np.array_equal(events.pred_original.to_numpy(), original.loc[events.row_key].pred_original.to_numpy())
    assert np.allclose(events.pred_changed - events.pred_original, events.prediction_delta, equal_nan=True, rtol=0, atol=1e-14)
    changed = events[events.item1.ne('')]
    for item in ITEMS:
        sub = changed[changed.item1.eq(item)]
        assert np.array_equal(sub.original_value1.to_numpy(), source.loc[sub.participant_id, item].to_numpy())
        assert np.array_equal(sub.changed_value1.to_numpy(), (sub.original_value1 + sub.step1).to_numpy())
    joint = changed[changed.item2.ne('')]
    assert np.array_equal(joint.original_value2.to_numpy(), source.loc[joint.participant_id, LKS].to_numpy())
    assert np.array_equal(joint.changed_value2.to_numpy(), (joint.original_value2 + joint.step2).to_numpy())
    assert changed.loc[changed.boundary_valid, 'pred_changed'].notna().all()
    assert changed.loc[~changed.boundary_valid, 'pred_changed'].isna().all()
    grouped = events.groupby(['participant_id', 'setting']).prediction_delta.mean()
    saved = person[person.scope.eq('pooled')].set_index(['participant_id', 'setting']).prediction_delta
    assert np.allclose(grouped, saved.reindex(grouped.index), equal_nan=True, rtol=0, atol=1e-14)
    assert neighbors.groupby(['participant_id', 'context']).size().eq(K).all()
    assert not neighbors.duplicated(['participant_id', 'context', 'training_neighbor_id']).any()
    train_folds = source.loc[neighbors.training_neighbor_id, 'fold'].to_numpy()
    assert np.all(train_folds != neighbors.fold.to_numpy())
    assert np.all(source.loc[neighbors.participant_id, 'fold'].to_numpy() == neighbors.fold.to_numpy())
    for item in ITEMS:
        nn = neighbors[neighbors.context.eq(item)].copy()
        nn['neighbor_value'] = source.loc[nn.training_neighbor_id, item].to_numpy()
        counts = nn.groupby(['participant_id', 'neighbor_value']).size().unstack(fill_value=0).reindex(columns=range(11), fill_value=0)
        ss = support[support.context.eq(item)].set_index('participant_id').reindex(counts.index)
        assert np.array_equal(counts.to_numpy(), ss[[f'neighbor_count_{v}' for v in range(11)]].to_numpy())
        assert np.allclose(nn.groupby('participant_id').distance.max().reindex(ss.index), ss.d100, rtol=0, atol=1e-14)
        for threshold in COUNTS:
            expect = ss.original_value.between(1, 9) & ss.radius_supported & (ss[['count_step_-1','count_step_0','count_step_1']].min(axis=1) >= threshold)
            assert np.array_equal(expect, ss[f'primary_eligible_m{threshold}'])
    matrix = np.load(dest / 'whatif_shared_bootstrap_weights.npz')['weights']
    ids = pd.read_parquet(dest / 'whatif_bootstrap_participants.parquet').participant_id
    assert matrix.shape == (B, len(p)) and np.all(matrix.sum(axis=1) == len(p)) and np.all(matrix >= 0)
    assert (matrix > 1).any(), 'Multiplicity lost in bootstrap weights'
    contrasts = pd.read_parquet(dest / 'whatif_participant_contrasts.parquet')
    primary = pd.read_csv(tables / 'whatif_primary.csv')
    primary = primary[primary.scope.eq('pooled')]
    deviations = []; max_ci_error = 0.
    for row in primary.itertuples():
        part = contrasts[(contrasts.analysis == 'primary') & (contrasts.context == row.context) &
                         (contrasts.contrast == f'{row.contrast:+d}')].set_index('participant_id')
        vals = part.participant_delta.reindex(ids).to_numpy()
        ok = np.isfinite(vals)
        assert int(ok.sum()) == row.n_eligible
        if ok.sum() < MIN_N:
            continue
        reps = matrix[:, ok] @ vals[ok] / matrix[:, ok].sum(axis=1)
        point = vals[ok].mean(); se = reps.std(ddof=1)
        assert abs(point - row.estimate) < 1e-12
        endpoints = np.quantile(reps, [.025,.975])
        max_ci_error = max(max_ci_error, float(np.max(np.abs(endpoints - [row.ci_low,row.ci_high]))))
        assert abs(se - row.bootstrap_se) < 1e-12
        if se > 1e-14:
            deviations.append(np.abs((reps - point) / se))
        else:
            assert np.max(np.abs(reps - point)) <= 1e-12
    critical = np.quantile(np.vstack(deviations).max(axis=0), .95) if deviations else 0.
    assert max_ci_error < 1e-12
    assert np.max(np.abs(primary.simultaneous_critical - critical)) < 1e-12
    report = dict(status='passed', event_cache_rows=len(events), participant_cache_rows=len(person),
                  support_rows=len(support), neighbor_rows=len(neighbors), changed_input_values_match_source=True,
                  invalid_boundary_predictions_missing=True, event_to_person_aggregation_matches=True,
                  training_neighbors_unique_and_no_test_overlap=True, local_counts_reconstructed=True,
                  common_eligible_sets_reconstructed=True, bootstrap_multiplicity_preserved=True,
                  primary_means_CIs_and_simultaneous_critical_reconstructed=True,
                  max_pointwise_ci_reconstruction_error=max_ci_error)
    dump(report, reports / 'whatif_validation.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = args.output_dir or root / 'outputs/whatif'
    reports, tables, dest = [out / x for x in ['reports', 'tables', 'data']]
    for d in (reports, tables, dest):
        d.mkdir(parents=True, exist_ok=True)
    plan_path = REPO / 'config/whatif.json'
    if not plan_path.exists():
        raise RuntimeError('Missing config/whatif.json; refusing to infer or change the study support rules.')
    plan = json.loads(plan_path.read_text())
    assert plan['primary']['items'] == ITEMS and plan['support']['k'] == K
    assert plan['bootstrap']['draws'] == B and plan['bootstrap']['seed'] == SEED
    assert plan['support']['min_count_primary'] == 10 and plan['support']['min_count_sensitivity'] == [5, 20]
    assert plan['support']['minimum_eligible_people_for_effect_reporting'] == MIN_N
    start = time.time()
    files = [DATA_DIR/'participants.parquet', DATA_DIR/'events.parquet', root/'outputs/core/prediction/m2_oof_predictions.parquet', root/'outputs/core/prediction/m2_fold_membership.parquet']
    files += sorted((root/'outputs/core/prediction/models').glob('*.joblib'))
    plan['source_hashes'] = {relative_label(p,root):digest(p) for p in files}
    p, data, oof = read_data(root)
    with threadpool_limits(limits=2):
        bundles, baseline = baseline_validation(root, data, oof, reports)
        baseline.to_parquet(dest / 'whatif_baseline_oof_reproduction.parquet', index=False)
        print('BASELINE OOF REPRODUCTION PASSED BEFORE PERTURBATION', flush=True)
        supports, calibration, neighbors, params = [], [], [], []
        for fold, bundle in bundles.items():
            train_ids = set(data.set_index('row_key').loc[bundle['train_row_keys'], 'participant_id'])
            test_ids = set(data.set_index('row_key').loc[bundle['test_row_keys'], 'participant_id'])
            tr = p[p.participant_id.isin(train_ids)].copy()
            te = p[p.participant_id.isin(test_ids)].copy()
            assert te.fold.eq(str(fold)).all() and not tr.fold.eq(str(fold)).any()
            s, c, n, par = support_fold(tr, te, str(fold))
            supports.append(s); calibration.append(c); neighbors.append(n); params.extend(par)
            print(f'SUPPORT fold={fold} train_people={len(tr)} heldout_people={len(te)}', flush=True)
        support = pd.concat(supports, ignore_index=True)
        assert not support.duplicated(['participant_id', 'context']).any()
        support.to_parquet(dest / 'whatif_participant_support.parquet', index=False)
        pd.concat(calibration, ignore_index=True).to_parquet(dest / 'whatif_training_radius_calibration.parquet', index=False)
        pd.concat(neighbors, ignore_index=True).to_parquet(dest / 'whatif_training_neighbors.parquet', index=False)
        pd.DataFrame(params).to_csv(tables / 'whatif_gower_training_parameters.csv', index=False)
        person, event_rows = generate_predictions(data, bundles, baseline, support, dest)
        results = summarize_results(p, support, person, tables, reports, dest)
        validate_outputs(p, support, person, dest, tables, reports)
    unchanged = {rel: digest(Path(rel) if Path(rel).is_absolute() else root / rel) == sha for rel, sha in plan['source_hashes'].items()}
    assert all(unchanged.values())
    versions = {}
    for package in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'joblib', 'pyarrow', 'threadpoolctl']:
        versions[package] = importlib.metadata.version(package)
    files = [f for folder in (dest, tables) for f in folder.glob('whatif_*') if f.is_file()]
    files += [reports / 'whatif_oof_reproduction.json', reports / 'whatif_simultaneous_intervals.json', reports / 'whatif_validation.json']
    manifest = dict(status='passed', elapsed_seconds=time.time() - start, plan_sha256=digest(plan_path),
        script_sha256=digest(Path(__file__)), plan_addenda_sha256={f.name: digest(f) for f in reports.glob('analysis_plan_addendum_*.json')}, python=platform.python_version(), package_versions=versions,
        n_participants=len(p), n_observed_events=len(data), n_event_prediction_cache_rows=event_rows,
        n_participant_prediction_cache_rows=len(person), n_contrast_rows=len(results),
        source_files_unchanged=unchanged,
        outputs={str(f.relative_to(out)):dict(sha256=digest(f), bytes=f.stat().st_size) for f in files},
        interpretation='Predictive model sensitivity, conditional on fixed held-out models and observed stimuli. No causal inference.',
        support_rule_loaded_before_prediction=True, manuscript_or_figures_modified=False)
    dump(manifest, reports / 'whatif_manifest.json')
    print(json.dumps({k:manifest[k] for k in ['status','elapsed_seconds','n_participants','n_observed_events','n_event_prediction_cache_rows','n_contrast_rows']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
