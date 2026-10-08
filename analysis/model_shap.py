"""Round 2: positive operated video ratings, all countries, held-out TreeSHAP.

Questionnaire identities are verified; physical stimulus links are unresolved.
All covariates are explicitly whitelisted pre-task attributes. Video-risk zeros
are missing operations, whereas zero-valued pre-task items remain valid.
"""
from __future__ import annotations
from settings import WORKSPACE as ROOT, DATA_DIR, QUESTIONNAIRE, REPO, relative_label
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "2")

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits


SEED = 20261001
NUMERIC = ["age", "driving_exp_years", "trust_pre_overall", "trust_pre_acc", "trust_pre_lks",
           "accept_pre_delegate", "accept_pre_monitor_raw", "accept_pre_distract"]
CATEGORICAL = ["gender", "education", "country", "own_car", "drive_freq_12m", "km_life", "km_12m",
               "drive_type", "drive_style", "know_ad", "acc_years", "acc_dist", "lks_years", "lks_dist",
               "acclks_years", "acclks_dist"]
IDENTITY = ["scenario", "event_key"]
DISPLAY = {
    "age": "Age", "driving_exp_years": "Driving experience", "trust_pre_overall": "Initial overall trust",
    "trust_pre_acc": "Initial ACC trust", "trust_pre_lks": "Initial LKS trust",
    "accept_pre_delegate": "Initial delegation willingness", "accept_pre_monitor_raw": "Initial need to monitor",
    "accept_pre_distract": "Initial distraction willingness", "gender": "Gender", "education": "Education",
    "country": "Reported country", "own_car": "Car ownership", "drive_freq_12m": "Driving frequency",
    "km_life": "Lifetime driving distance", "km_12m": "Recent driving distance", "drive_type": "Typical round-trip distance",
    "drive_style": "Self-described driving style", "know_ad": "Self-reported automation knowledge", "acc_years": "ACC use duration",
    "acc_dist": "ACC use distance", "lks_years": "LKS use duration", "lks_dist": "LKS use distance",
    "acclks_years": "Combined ACC/LKS use duration", "acclks_dist": "Combined ACC/LKS use distance",
    "scenario": "Questionnaire block", "event_key": "Questionnaire stimulus identity",
}
GBM_FIXED = dict(objective="regression", learning_rate=.05, n_jobs=2, random_state=SEED,
                 verbosity=-1, deterministic=True, force_col_wise=True)
GBM_GRID = [dict(n_estimators=[200], num_leaves=[15], max_depth=[4], min_child_samples=[50], reg_lambda=[1.0]),
            dict(n_estimators=[350], num_leaves=[31], max_depth=[6], min_child_samples=[80], reg_lambda=[5.0])]

# Generic preprocessing, split and cluster-bootstrap utilities are inlined for
# independent reproduction. Their behavior matches the reviewed round-1 
def stable_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_preprocessor(numeric, categorical):
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ], verbose_feature_names_out=True)


def assignments(values, n_splits, seed):
    """Random fixed group partition, independent of outcomes and input ordering."""
    groups = np.asarray(sorted(pd.unique(values).tolist()), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    return {value: int(i % n_splits) for i, value in enumerate(groups)}


def make_splits(data):
    pid_fold = data.participant_id.map(assignments(data.participant_id, 5, SEED)).to_numpy()
    event_map = {}
    for i, (_, sub) in enumerate(data.groupby("scenario", sort=True)):
        event_map.update(assignments(sub.event_key, 5, SEED + 100 + i))
    event_fold = data.event_key.map(event_map).to_numpy()
    splits = []
    for f in range(5):
        splits.append(("participant_5fold", str(f), np.flatnonzero(pid_fold != f), np.flatnonzero(pid_fold == f)))
    for country in sorted(data.country.unique()):
        test = (data.country == country).to_numpy()
        splits.append(("leave_country_out", str(country), np.flatnonzero(~test), np.flatnonzero(test)))
    for f in range(5):
        splits.append(("event_group_5fold", str(f), np.flatnonzero(event_fold != f), np.flatnonzero(event_fold == f)))
    for fp in range(5):
        for fe in range(5):
            train = (pid_fold != fp) & (event_fold != fe)
            test = (pid_fold == fp) & (event_fold == fe)
            splits.append(("crossed_participant_event_5x5", f"p{fp}_e{fe}", np.flatnonzero(train), np.flatnonzero(test)))
    return splits, pid_fold, event_fold


def audit_split(data, protocol, fold, tr, te):
    train, test = data.iloc[tr], data.iloc[te]
    overlap = {
        "participant_overlap": len(set(train.participant_id) & set(test.participant_id)),
        "event_overlap": len(set(train.event_key) & set(test.event_key)),
        "country_overlap": len(set(train.country) & set(test.country)),
        "row_overlap": len(set(train.row_key) & set(test.row_key)),
    }
    assert overlap["row_overlap"] == 0
    if protocol in ("participant_5fold", "leave_country_out", "crossed_participant_event_5x5"):
        assert overlap["participant_overlap"] == 0
    if protocol in ("event_group_5fold", "crossed_participant_event_5x5"):
        assert overlap["event_overlap"] == 0
    if protocol == "leave_country_out":
        assert overlap["country_overlap"] == 0
    return {"protocol": protocol, "fold": fold, "n_train": len(tr), "n_test": len(te), **overlap}


def person_cluster_ci(frame, n_boot=1000):
    """Percentile CIs conditional on fitted OOF models and this stimulus set.

    Repeated participants are given their bootstrap multiplicity (never isin).
    These CIs do not propagate model-refit or stimulus-sampling uncertainty.
    """
    a = frame.assign(_sse=(frame.actual-frame.pred)**2,
                     _sae=np.abs(frame.actual-frame.pred),
                     _y2=frame.actual**2)
    g = a.groupby("participant_id").agg(n=("actual", "size"), sy=("actual", "sum"),
                                          sy2=("_y2", "sum"), sse=("_sse", "sum"), sae=("_sae", "sum"))
    vals = g.to_numpy()
    rng = np.random.default_rng(SEED)
    sums = np.asarray([vals[rng.integers(0, len(vals), len(vals))].sum(axis=0) for _ in range(n_boot)])
    n, sy, sy2, sse, sae = sums.T
    metrics = {"r2": 1-sse/(sy2-sy**2/n), "rmse": np.sqrt(sse/n), "mae": sae/n}
    return {f"{metric}_ci_{bound}": float(np.quantile(values, q))
            for metric, values in metrics.items() for bound, q in [("low", .025), ("high", .975)]}


def increment_ci(wide, n_boot=1000):
    """Paired participant bootstrap of added-person R², using the same draws."""
    a = wide.assign(_delta_sse=(wide.actual-wide.stimulus_only)**2-(wide.actual-wide.stimulus_plus_person)**2,
                    _y2=wide.actual**2)
    g = a.groupby("participant_id").agg(n=("actual", "size"), sy=("actual", "sum"),
                                         sy2=("_y2", "sum"), delta_sse=("_delta_sse", "sum"))
    values = g.to_numpy()
    rng = np.random.default_rng(SEED)
    sums = np.asarray([values[rng.integers(0, len(values), len(values))].sum(axis=0) for _ in range(n_boot)])
    n, sy, sy2, difference = sums.T
    increment = difference/(sy2-sy**2/n)
    return {"person_increment_r2_ci_low": float(np.quantile(increment, .025)),
            "person_increment_r2_ci_high": float(np.quantile(increment, .975))}


def summarize_predictions(predictions, group_columns, ci=True):
    rows = []
    for keys, frame in predictions.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row.update(n=len(frame), n_participants=frame.participant_id.nunique(),
                   rmse=float(np.sqrt(mean_squared_error(frame.actual, frame.pred))),
                   mae=float(mean_absolute_error(frame.actual, frame.pred)),
                   r2=float(r2_score(frame.actual, frame.pred)))
        rhos = [float(spearmanr(g.actual, g.pred).statistic) for _, g in frame.groupby("participant_id")
                if len(g) >= 3 and g.actual.nunique() > 1 and g.pred.nunique() > 1]
        row["within_participant_rho"] = float(np.mean(rhos)) if rhos else np.nan
        if ci:
            row.update(person_cluster_ci(frame))
        rows.append(row)
    return pd.DataFrame(rows)

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def source_feature(encoded, names):
    plain = encoded.split("__", 1)[-1]
    if encoded.startswith("num__"):
        return plain
    for name in sorted(names, key=len, reverse=True):
        if plain.startswith(name + "_"):
            return name
    raise ValueError(f"Unknown encoded feature: {encoded}")


def feature_family(name):
    if name in IDENTITY:
        return "Questionnaire identity"
    if name == "country":
        return "Reported country"
    if name.startswith("trust_pre"):
        return "Initial trust"
    if name.startswith("accept_pre"):
        return "Initial acceptance items"
    if name in ["acc_years", "acc_dist", "lks_years", "lks_dist", "acclks_years", "acclks_dist", "know_ad"]:
        return "Automation experience"
    if name in ["age", "driving_exp_years", "gender", "education"]:
        return "Demographics and experience"
    return "Driving history and habits"


def load_data(data_dir):
    p = pd.read_parquet(data_dir / "participants.parquet")
    e = pd.read_parquet(data_dir / "events.parquet")
    assert p.participant_id.is_unique and e.event_row_id.is_unique
    expected = NUMERIC + CATEGORICAL
    missing = [c for c in expected if c not in p]
    if missing:
        raise ValueError(f"Missing pre-task variables: {missing}")
    # Use the participant table as the single authoritative covariate source.
    event_columns = [c for c in ["event_row_id", "participant_id", "scenario", "event_key", "questionnaire_block",
                                "questionnaire_event_id", "event_id", "risk_event_mean", "n_valid_windows",
                                "n_zero_no_operation", "n_windows_expected", "event_risk_available"] if c in e]
    data = e[event_columns].merge(p[["participant_id"] + expected], on="participant_id", validate="m:1")
    data = data.rename(columns={"event_row_id": "row_key"})
    data["event_risk_available"] = data.risk_event_mean.notna()
    coverage = data.groupby("country", dropna=False).agg(participants=("participant_id", "nunique"),
        assigned_events=("row_key", "size"), observed_events=("event_risk_available", "sum"),
        valid_windows=("n_valid_windows", "sum"), zero_no_operation=("n_zero_no_operation", "sum")).reset_index()
    analysis = data[data.event_risk_available].copy()
    assert analysis.risk_event_mean.gt(0).all() and analysis.risk_event_mean.le(10).all()
    for c in NUMERIC:
        analysis[c] = pd.to_numeric(analysis[c], errors="coerce").astype(float)
    for c in CATEGORICAL + IDENTITY:
        analysis[c] = analysis[c].fillna("__missing__").astype(str)
    analysis = analysis.sort_values(["participant_id", "scenario", "event_key"]).reset_index(drop=True)
    assert analysis.row_key.is_unique
    manifest = dict(n_source_participants=len(p), n_model_participants=int(analysis.participant_id.nunique()),
        n_assigned_events=len(data), n_observed_events=len(analysis), n_all_missing_events=int((~data.event_risk_available).sum()),
        countries_in_model=sorted(analysis.country.unique().tolist()), country_n_threshold=None,
        outcome="Mean of valid video ratings greater than zero within an assigned event; zero means no operation, not no risk",
        event_weighting="Equal observed participant-event row weights; missing-window count is not a predictor",
        zero_prepost_items="Valid values retained; no zero recoding of trust or acceptance",
        person_numeric=NUMERIC, person_categorical=CATEGORICAL, identity=IDENTITY,
        physical_mapping="unresolved; no physical-feature join in primary models",
        sample_rule="All completed, licensed participants; no duration or country-size threshold in the primary sample",
        input_hashes={name: stable_digest(data_dir / f"{name}.parquet") for name in ["participants", "events", "windows"]})
    return p, analysis, coverage, manifest


def fit_nested(train, model, group_col, drop=()):
    numeric = [c for c in NUMERIC if c not in drop]
    cats = [c for c in CATEGORICAL + IDENTITY if c not in drop]
    pre = make_preprocessor(numeric, cats)
    estimator = Ridge(solver="cholesky") if model == "Ridge" else lgb.LGBMRegressor(**GBM_FIXED)
    pipeline = Pipeline([("pre", pre), ("model", estimator)])
    grid = {"model__alpha": [.1, 10., 100.]} if model == "Ridge" else [{f"model__{k}": v for k, v in config.items()} for config in GBM_GRID]
    if group_col == "participant_and_event":
        pf = train.participant_id.map(assignments(train.participant_id,3,SEED+11)).to_numpy()
        event_assignment={}
        for i,(_,block) in enumerate(train.groupby("scenario",sort=True)):
            event_assignment.update(assignments(block.event_key,3,SEED+23+i))
        ef=train.event_key.map(event_assignment).to_numpy()
        inner=[]
        for fp in range(3):
            for fe in range(3):
                tr=np.flatnonzero((pf!=fp)&(ef!=fe))
                va=np.flatnonzero((pf==fp)&(ef==fe))
                assert len(tr)>0 and len(va)>0
                assert not set(train.iloc[tr].participant_id)&set(train.iloc[va].participant_id)
                assert not set(train.iloc[tr].event_key)&set(train.iloc[va].event_key)
                inner.append((tr,va))
        fit_groups=None
    else:
        inner = GroupKFold(n_splits=3)
        fit_groups=train[group_col]
    search = GridSearchCV(pipeline, grid, scoring="neg_mean_squared_error", cv=inner, n_jobs=1,
                          error_score="raise", refit=True, return_train_score=False)
    search.fit(train, train.risk_event_mean, groups=fit_groups)
    return search.best_estimator_, {"best_parameters": search.best_params_,
        "inner_group": group_col, "inner_folds":9 if group_col=="participant_and_event" else 3,
        "inner_cv_rmse": float(np.sqrt(-search.best_score_)),
        "candidate_mean_rmse": np.sqrt(-search.cv_results_["mean_test_score"]).tolist()}


def metadata_rows(test, protocol, fold, model, prediction):
    columns = ["row_key", "participant_id", "event_key", "scenario", "country", "n_valid_windows"]
    out = test[columns].copy()
    return out.assign(protocol=protocol, fold=str(fold), model=model,
                      actual=test.risk_event_mean.to_numpy(), pred=np.asarray(prediction))


def fit_all(data, out, tables, reports):
    predictions, audits, configs = [], [], []
    splits, pf, ef = make_splits(data)
    membership = data[["row_key", "participant_id", "event_key", "country", "scenario"]].copy()
    membership["participant_fold"] = pf
    membership["event_fold"] = ef
    membership.to_parquet(out / "m2_fold_membership.parquet", index=False)
    (out / "models").mkdir(exist_ok=True)
    for protocol, fold, tr, te in splits:
        start = time.monotonic()
        audits.append(audit_split(data, protocol, fold, tr, te))
        train, test = data.iloc[tr], data.iloc[te]
        train_means = train.groupby("event_key").risk_event_mean.mean()
        blocks = train.groupby("scenario").risk_event_mean.mean()
        ref = test.event_key.map(train_means).fillna(test.scenario.map(blocks)).fillna(train.risk_event_mean.mean())
        predictions.append(metadata_rows(test, protocol, fold, "Training event/block mean", ref))
        predictions.append(metadata_rows(test, protocol, fold, "Training mean", np.full(len(test), train.risk_event_mean.mean())))
        groups = inner_group_for(protocol)
        for model in ["Ridge", "LightGBM"]:
            pipe, settings = fit_nested(train, model, groups)
            predictions.append(metadata_rows(test, protocol, fold, model, pipe.predict(test)))
            configs.append(dict(protocol=protocol, fold=str(fold), model=model, **settings))
            if protocol == "participant_5fold" and model == "LightGBM":
                joblib.dump({"pipeline": pipe, "train_row_keys": train.row_key.tolist(), "test_row_keys": test.row_key.tolist(),
                             "settings": settings, "numeric": NUMERIC, "categorical": CATEGORICAL + IDENTITY},
                            out / "models" / f"m2_participant_fold_{fold}.joblib")
        print(f"FIT {protocol} {fold}: train={len(train)} test={len(test)} elapsed={time.monotonic()-start:.1f}s", flush=True)
    predictions = pd.concat(predictions, ignore_index=True)
    assert not predictions.duplicated(["protocol", "model", "row_key"]).any()
    assert predictions.groupby(["protocol", "model"]).size().eq(len(data)).all()
    assert np.isfinite(predictions.pred).all()
    predictions.to_parquet(out / "m2_oof_predictions.parquet", index=False)
    pd.DataFrame(audits).to_csv(tables / "m2_split_audit.csv", index=False)
    write_json(reports / "m2_fit_settings.json", configs)
    return fit_identity_comparators(data, out, tables, reports)


def fit_identity_comparators(data, out, tables, reports):
    """Matched same-algorithm references for attributing predictive increments."""
    existing = pd.read_parquet(out / "m2_oof_predictions.parquet")
    existing = existing[~existing.model.str.endswith(" identity only")].copy()
    predictions, configs = [existing], []
    splits, _, _ = make_splits(data)
    for protocol, fold, tr, te in splits:
        train, test = data.iloc[tr], data.iloc[te]
        group_col = inner_group_for(protocol)
        for model in ["Ridge", "LightGBM"]:
            pipe, settings = fit_nested(train, model, group_col, drop=NUMERIC+CATEGORICAL)
            predictions.append(metadata_rows(test,protocol,fold,f"{model} identity only",pipe.predict(test)))
            configs.append(dict(protocol=protocol,fold=str(fold),model=model,feature_set="identity_only",**settings))
        print(f"IDENTITY COMPARATOR {protocol} {fold}",flush=True)
    predictions = pd.concat(predictions,ignore_index=True)
    assert not predictions.duplicated(["protocol","model","row_key"]).any()
    assert predictions.groupby(["protocol","model"]).size().eq(len(data)).all()
    predictions.to_parquet(out / "m2_oof_predictions.parquet",index=False)
    write_json(reports / "m2_identity_fit_settings.json",configs)
    summarize(predictions,tables)
    return predictions


def inner_group_for(protocol):
    return {"leave_country_out":"country","event_group_5fold":"event_key",
            "crossed_participant_event_5x5":"participant_and_event"}.get(protocol,"participant_id")


def refit_crossed(data,out,tables,reports):
    """Target-matched nested crossed tuning, preserving all other OOF results."""
    protocol="crossed_participant_event_5x5"
    existing=pd.read_parquet(out/"m2_oof_predictions.parquet")
    predictions=[existing[existing.protocol!=protocol]]
    full_settings=[c for c in json.loads((reports/"m2_fit_settings.json").read_text()) if c["protocol"]!=protocol]
    identity_settings=[c for c in json.loads((reports/"m2_identity_fit_settings.json").read_text()) if c["protocol"]!=protocol]
    splits,_,_=make_splits(data)
    for current,fold,tr,te in splits:
        if current!=protocol:continue
        train,test=data.iloc[tr],data.iloc[te]
        ref=test.scenario.map(train.groupby("scenario").risk_event_mean.mean()).fillna(train.risk_event_mean.mean())
        predictions.append(metadata_rows(test,protocol,fold,"Training event/block mean",ref))
        predictions.append(metadata_rows(test,protocol,fold,"Training mean",np.full(len(test),train.risk_event_mean.mean())))
        for model in ["Ridge","LightGBM"]:
            for identity_only in [False,True]:
                drop=NUMERIC+CATEGORICAL if identity_only else ()
                pipe,settings=fit_nested(train,model,"participant_and_event",drop=drop)
                label=f"{model} identity only" if identity_only else model
                predictions.append(metadata_rows(test,protocol,fold,label,pipe.predict(test)))
                target=identity_settings if identity_only else full_settings
                target.append(dict(protocol=protocol,fold=str(fold),model=model,feature_set="identity_only" if identity_only else "identity_plus_person",**settings))
        print(f"CROSSED NESTED {fold}",flush=True)
    predictions=pd.concat(predictions,ignore_index=True)
    assert not predictions.duplicated(["protocol","model","row_key"]).any()
    assert predictions.groupby(["protocol","model"]).size().eq(len(data)).all()
    predictions.to_parquet(out/"m2_oof_predictions.parquet",index=False)
    write_json(reports/"m2_fit_settings.json",full_settings)
    write_json(reports/"m2_identity_fit_settings.json",identity_settings)
    summarize(predictions,tables)
    return predictions


def summarize(predictions, tables):
    metrics = summarize_predictions(predictions, ["protocol", "model"], ci=True)
    metrics.to_csv(tables / "m2_performance.csv", index=False)
    country = summarize_predictions(predictions[predictions.protocol == "leave_country_out"], ["country", "model"], ci=False)
    country["precision_note"] = "No country-specific confidence interval; small-country point estimates are unstable"
    country.to_csv(tables / "m2_country_holdout_performance.csv", index=False)
    # Equal participant-weight evaluation is a metric sensitivity, not a refit.
    weighted = []
    for (protocol, model), block in predictions.groupby(["protocol", "model"]):
        weights = 1 / block.groupby("participant_id").row_key.transform("size")
        weighted.append(dict(protocol=protocol, model=model, evaluation="equal_participant_weights_same_predictions",
            r2=r2_score(block.actual, block.pred, sample_weight=weights),
            rmse=np.sqrt(mean_squared_error(block.actual, block.pred, sample_weight=weights)),
            mae=mean_absolute_error(block.actual, block.pred, sample_weight=weights)))
    pd.DataFrame(weighted).to_csv(tables / "m2_participant_weighted_metrics.csv", index=False)
    increments=[]
    for protocol, block in predictions.groupby("protocol"):
        for model in ["Ridge","LightGBM"]:
            if f"{model} identity only" not in set(block.model):
                continue
            baseline = block[block.model==f"{model} identity only"][["row_key","participant_id","actual","pred"]].rename(columns={"pred":"stimulus_only"})
            person = block[block.model==model][["row_key","pred"]].rename(columns={"pred":"stimulus_plus_person"})
            wide = baseline.merge(person,on="row_key",validate="1:1")
            delta = r2_score(wide.actual,wide.stimulus_plus_person)-r2_score(wide.actual,wide.stimulus_only)
            increments.append(dict(protocol=protocol,model=model,n=len(wide),increment_r2=delta,
                contrast="Same algorithm and outer split; identity plus all pre-task attributes versus identity only",
                **increment_ci(wide)))
    pd.DataFrame(increments).to_csv(tables/"m2_person_increment.csv",index=False)
    return metrics


def sum_encoded(matrix, encoded_names, source_names):
    owner = [source_feature(n, source_names) for n in encoded_names]
    aggregation = np.asarray([[float(o == c) for c in source_names] for o in owner])
    return matrix @ aggregation, aggregation


def explain(data, out, tables, reports, sample_participants=20):
    source_names = NUMERIC + CATEGORICAL + IDENTITY
    all_contrib, pair_records, group_interactions, sens_preds, sens_importance = [], [], [], [], []
    audit = []
    lookup = data.set_index("row_key", drop=False)
    for fold in range(5):
        bundle = joblib.load(out / "models" / f"m2_participant_fold_{fold}.joblib")
        pipe = bundle["pipeline"]
        train, test = lookup.loc[bundle["train_row_keys"]], lookup.loc[bundle["test_row_keys"]]
        assert not set(train.participant_id) & set(test.participant_id)
        X = pipe.named_steps["pre"].transform(test)
        names = list(pipe.named_steps["pre"].get_feature_names_out())
        booster = pipe.named_steps["model"].booster_
        native = booster.predict(X, pred_contrib=True, num_threads=2)
        pred = booster.predict(X, num_threads=2)
        assert np.allclose(native.sum(axis=1), pred, atol=1e-8)
        grouped, aggregation = sum_encoded(native[:, :-1], names, source_names)
        frame = pd.DataFrame(grouped, columns=source_names)
        frame.columns = [f"shap__{c}" for c in source_names]
        frame["row_key"] = test.row_key.to_numpy()
        frame["participant_id"] = test.participant_id.to_numpy()
        frame["base_value"] = native[:, -1]
        frame["pred"] = pred
        frame["actual"] = test.risk_event_mean.to_numpy()
        frame["fold"] = fold
        for c in NUMERIC:
            frame[f"value__{c}"] = test[c].to_numpy()
        all_contrib.append(frame)
        # Sample participants first, then one observed event per questionnaire
        # block for each selected participant; retain keys and sample definition.
        rng = np.random.default_rng(SEED + fold)
        pids = rng.choice(test.participant_id.unique(), size=min(sample_participants, test.participant_id.nunique()), replace=False)
        selected = []
        for pid in pids:
            for _, group in test[test.participant_id == pid].groupby("scenario"):
                selected.append(rng.choice(group.index.to_numpy()))
        sampled = test.loc[selected]
        Xi = pipe.named_steps["pre"].transform(sampled)
        explainer = shap.TreeExplainer(pipe.named_steps["model"], feature_perturbation="tree_path_dependent", model_output="raw")
        interactions = np.asarray(explainer.shap_interaction_values(Xi))
        atomic = booster.predict(Xi, pred_contrib=True, num_threads=2)[:, :-1]
        assert interactions.ndim == 3
        row_error = float(np.max(np.abs(interactions.sum(axis=2) - atomic)))
        sym_error = float(np.max(np.abs(interactions - interactions.transpose(0, 2, 1))))
        assert row_error < 1e-7 and sym_error < 1e-7
        # Sum atomic interactions within original-variable pairs. This is a
        # displayed aggregation, not a redefinition of coalition-level SHAP.
        grouped_inter = np.einsum("if,nij,jg->nfg", aggregation, interactions, aggregation, optimize=True)
        group_interactions.append(dict(fold=fold, n=len(sampled), mean_abs_pair=2*np.abs(grouped_inter).mean(axis=0)))
        for i, a in enumerate(NUMERIC):
            for j, b in enumerate(NUMERIC):
                if i >= j:
                    continue
                # Numeric variables each occupy one model dimension. This is
                # the true unordered pair contribution 2*Phi_ij, not coloring.
                block = sampled[["row_key", "participant_id", "scenario", "country"]].copy()
                block["fold"], block["feature_a"], block["feature_b"] = fold, a, b
                block["value_a"], block["value_b"] = sampled[a].to_numpy(), sampled[b].to_numpy()
                block["pair_contribution"] = 2*grouped_inter[:, i, j]
                pair_records.append(block.reset_index(drop=True))
        audit.append(dict(fold=fold, sample_participants=len(pids), sample_rows=len(sampled),
                          encoded_features=len(names), interaction_row_sum_error=row_error, interaction_symmetry_error=sym_error))
        # Age/experience attribution sensitivity: keep the selected training-only
        # tree settings; refit without one correlated covariate on the same fold.
        for omitted in ["age", "driving_exp_years"]:
            num = [c for c in NUMERIC if c != omitted]
            params = {k.replace("model__", ""): v for k, v in bundle["settings"]["best_parameters"].items()}
            model = Pipeline([("pre", make_preprocessor(num, CATEGORICAL+IDENTITY)),
                              ("model", lgb.LGBMRegressor(**GBM_FIXED, **params))])
            model.fit(train, train.risk_event_mean)
            XX = model.named_steps["pre"].transform(test)
            nn = list(model.named_steps["pre"].get_feature_names_out())
            cc = model.named_steps["model"].booster_.predict(XX, pred_contrib=True, num_threads=2)
            pp = cc.sum(axis=1)
            sens_preds.append(metadata_rows(test, "participant_5fold", fold, f"LightGBM_without_{omitted}", pp))
            grouped_s, _ = sum_encoded(cc[:, :-1], nn, num+CATEGORICAL+IDENTITY)
            for k, c in enumerate(num+CATEGORICAL+IDENTITY):
                sens_importance.append(dict(fold=fold, specification=f"without_{omitted}", feature=c,
                                            mean_abs_shap=float(np.abs(grouped_s[:, k]).mean()), n=len(test)))
        print(f"EXPLAIN fold {fold}: OOF={len(test)} interaction_sample={len(sampled)} dimensions={len(names)}", flush=True)
    all_contrib = pd.concat(all_contrib, ignore_index=True)
    assert all_contrib.row_key.is_unique and set(all_contrib.row_key) == set(data.row_key)
    assert np.allclose(all_contrib[[f"shap__{c}" for c in source_names]].sum(axis=1)+all_contrib.base_value, all_contrib.pred, atol=1e-8)
    all_contrib.to_parquet(out / "m2_oof_shap.parquet", index=False)
    pair_df = pd.concat(pair_records, ignore_index=True)
    assert not pair_df.duplicated(["row_key", "feature_a", "feature_b"]).any()
    pair_df.to_parquet(out / "m2_sampled_true_interactions.parquet", index=False)
    weights = np.array([g["n"] for g in group_interactions])
    mean_pairs = np.average(np.stack([g["mean_abs_pair"] for g in group_interactions]), axis=0, weights=weights)
    pairs = [dict(feature_a=a, feature_b=b, mean_abs_pair_contribution=float(mean_pairs[i,j]),
                  definition="mean absolute 2*Phi_ij; one-hot dimensions summed by original variable", sample_rows=int(weights.sum()))
             for i,a in enumerate(source_names) for j,b in enumerate(source_names) if i<j]
    pd.DataFrame(pairs).sort_values("mean_abs_pair_contribution", ascending=False).to_csv(tables / "m2_global_pair_interactions.csv", index=False)
    importance = []
    for c in source_names:
        series = all_contrib[f"shap__{c}"]
        fold_abs = all_contrib.groupby("fold")[f"shap__{c}"].apply(lambda x: np.abs(x).mean())
        importance.append(dict(feature=c, family=feature_family(c), mean_abs_shap=float(series.abs().mean()),
                               fold_mean_abs_sd=float(fold_abs.std())))
    pd.DataFrame(importance).sort_values("mean_abs_shap", ascending=False).to_csv(tables / "m2_shap_global_importance.csv", index=False)
    families = []
    for fam in sorted(set(map(feature_family,source_names))):
        cols = [f"shap__{c}" for c in source_names if feature_family(c)==fam]
        families.append(dict(family=fam, mean_abs_grouped_shap=float(all_contrib[cols].sum(axis=1).abs().mean())))
    pd.DataFrame(families).to_csv(tables / "m2_shap_family_importance.csv", index=False)
    sens = pd.concat(sens_preds, ignore_index=True)
    sens.to_parquet(out / "m2_age_experience_sensitivity_predictions.parquet", index=False)
    summarize_predictions(sens, ["protocol","model"],ci=True).to_csv(tables / "m2_age_experience_sensitivity_metrics.csv",index=False)
    pd.DataFrame(sens_importance).to_csv(tables / "m2_age_experience_sensitivity_attribution.csv",index=False)
    write_json(reports / "m2_interaction_validation.json",dict(shap_version=shap.__version__,method="TreeSHAP tree_path_dependent raw",
        samples_per_fold=audit, pair_convention="2*off_diagonal_Phi; diagonal is main effect",
        feature_grouping="Categorical source-variable pairs aggregate atomic one-hot interactions; numeric pairs are direct TreeSHAP interactions",
        heldout=True, inference="descriptive model attribution, not causal or psychological mechanism"))
    return all_contrib, pair_df


def savefig(fig, figures, name):
    fig.savefig(figures / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_panel(fig,axes,figures,name):
    """Export a vector/raster panel from a multi-panel review figure."""
    fig.canvas.draw()
    boxes=[ax.get_tightbbox(fig.canvas.get_renderer()) for ax in axes]
    box=Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).expanded(1.035,1.055)
    fig.savefig(figures/f"{name}.png",dpi=200,bbox_inches=box)
    fig.savefig(figures/f"{name}.pdf",bbox_inches=box)


def plot_profiles(data, out, tables, figures):
    """Adapt the condition-combination view of SafeTraffic Copilot Fig. 6.

    The cut-points are descriptive display choices, not validated thresholds.
    Signed contributions and absolute attribution magnitudes are kept separate.
    Every explanation is from a model that held out that participant.
    """
    sv = pd.read_parquet(out / "m2_oof_shap.parquet")
    keys = ["row_key", "country", "trust_pre_acc", "trust_pre_lks", "driving_exp_years"]
    display = sv.merge(data[keys], on="row_key", validate="1:1")
    families = sorted(set(map(feature_family, NUMERIC + CATEGORICAL + IDENTITY)))
    family_labels={"Questionnaire identity":"Stimulus identity","Reported country":"Reported country",
        "Initial trust":"Initial trust","Initial acceptance items":"Initial acceptance",
        "Automation experience":"Automation experience","Demographics and experience":"Demographics / experience",
        "Driving history and habits":"Driving habits"}
    for fam in families:
        cols = [f"shap__{c}" for c in NUMERIC + CATEGORICAL + IDENTITY if feature_family(c) == fam]
        display[f"family__{fam}"] = display[cols].sum(axis=1)
    condition_cols = ["ACC trust ≤ 5", "LKS trust ≤ 5", "Driving experience ≤ 10 years"]
    eligible = display[["trust_pre_acc", "trust_pre_lks", "driving_exp_years"]].notna().all(axis=1)
    profiles = display[eligible].copy()
    profiles[condition_cols[0]] = profiles.trust_pre_acc.le(5)
    profiles[condition_cols[1]] = profiles.trust_pre_lks.le(5)
    profiles[condition_cols[2]] = profiles.driving_exp_years.le(10)
    profiles["condition_code"] = profiles[condition_cols].astype(int).astype(str).agg("".join, axis=1)
    rows = []
    rng = np.random.default_rng(SEED)
    for code, block in profiles.groupby("condition_code"):
        row = dict(condition_code=code, n_participants=block.participant_id.nunique(), n_events=len(block),
                   mean_prediction=block.pred.mean(), mean_observed=block.actual.mean(), mean_base=block.base_value.mean())
        for col in condition_cols:
            row[col] = bool(block[col].iloc[0])
        for fam in families:
            row[f"signed__{fam}"] = block[f"family__{fam}"].mean()
            row[f"magnitude__{fam}"] = block[f"family__{fam}"].abs().mean()
        cluster = block.groupby("participant_id").agg(n=("row_key", "size"), pred_sum=("pred", "sum"), actual_sum=("actual", "sum"))
        # Resampling cluster multiplicities is essential; no deduplication.
        if len(cluster) >= 5:
            draw = rng.multinomial(len(cluster), np.full(len(cluster), 1/len(cluster)), size=1000)
            denominator = draw @ cluster.n.to_numpy()
            for quantity, column in [("prediction", "pred_sum"), ("observed", "actual_sum")]:
                boot = (draw @ cluster[column].to_numpy()) / denominator
                row[f"{quantity}_ci_low"], row[f"{quantity}_ci_high"] = np.quantile(boot, [.025, .975])
        else:
            for quantity in ["prediction", "observed"]:
                row[f"{quantity}_ci_low"] = row[f"{quantity}_ci_high"] = np.nan
        assert np.isclose(row["mean_base"] + sum(row[f"signed__{f}"] for f in families), row["mean_prediction"])
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("mean_prediction").reset_index(drop=True)
    summary["profile"] = [f"C{i+1}" for i in range(len(summary))]
    mapping = dict(zip(summary.condition_code, summary.profile))
    profiles["profile"] = profiles.condition_code.map(mapping)
    summary.to_csv(tables / "m2_profile_attributions.csv", index=False)
    profiles[["row_key", "participant_id", "condition_code", "profile"]+condition_cols].to_parquet(out / "m2_profile_membership.parquet", index=False)

    plt.rcParams.update({"font.size": 6.5})
    fig = plt.figure(figsize=(7.08, 4.9))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.1], height_ratios=[1.1, 1], hspace=.5, wspace=.95)
    fig.subplots_adjust(left=.16,right=.96,top=.9,bottom=.13)
    ax_mean = fig.add_subplot(grid[0, 0])
    ax_condition = fig.add_subplot(grid[1, 0], sharex=ax_mean)
    ax_signed = fig.add_subplot(grid[0, 1])
    ax_importance = fig.add_subplot(grid[1, 1])
    positions = np.arange(len(summary))
    for quantity, marker, shift, color, label in [("prediction", "o", -.09, "#397b98", "Predicted"),
                                                ("observed", "s", .09, "#444444", "Observed")]:
        x = positions + shift
        ax_mean.vlines(x, summary[f"{quantity}_ci_low"], summary[f"{quantity}_ci_high"], color=color, lw=1.0)
        ax_mean.scatter(x, summary[f"mean_{quantity if quantity == 'prediction' else 'observed'}"], marker=marker,
                        s=14, color=color, facecolors="white" if quantity=="observed" else color, label=label, zorder=3)
    ax_mean.set_xticks(positions, summary.profile)
    ax_mean.set_ylabel("Mean operated risk rating")
    ax_mean.set_title("a  Profiles ordered by predicted risk", loc="left", fontsize=7)
    ax_mean.legend(frameon=False, fontsize=6, loc="upper left")
    ax_mean.tick_params(axis="x",labelbottom=False)
    ax_mean.spines[["top", "right"]].set_visible(False)
    for y, condition in enumerate(condition_cols):
        for x, active in enumerate(summary[condition]):
            ax_condition.scatter(x, y, s=25, color="#202020" if active else "#dedede", zorder=3)
    for x, row in summary.iterrows():
        selected = np.flatnonzero(row[condition_cols].to_numpy(dtype=bool))
        if len(selected)>1:
            ax_condition.plot([x,x], [selected.min(), selected.max()], color="#202020", lw=1)
    ax_condition.set_yticks(range(3), ["ACC trust ≤ 5","LKS trust ≤ 5","Experience ≤ 10 y"],fontsize=6)
    ax_condition.set_xticks(positions, [f"{r.profile}\n{r.n_participants}" for _, r in summary.iterrows()], fontsize=5.5)
    ax_condition.set_xlim(-.6, len(summary)-.4);ax_condition.set_ylim(2.7,-.7)
    ax_condition.set_title("c  Conditions (n participants below)",loc="left",fontsize=7)
    ax_condition.tick_params(axis="both", length=0)
    ax_condition.spines[["top","right","bottom","left"]].set_visible(False)
    signed = summary[[f"signed__{f}" for f in families]].to_numpy().T
    lim = max(np.abs(signed).max(),1e-5)
    im = ax_signed.imshow(signed, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),aspect="auto")
    ax_signed.set_yticks(range(len(families)), [family_labels[f] for f in families], fontsize=5.5)
    ax_signed.set_xticks(positions, summary.profile)
    ax_signed.set_title("b  Mean signed contributions",loc="left",fontsize=7)
    cb = fig.colorbar(im, ax=ax_signed, shrink=.9, pad=.03)
    cb.set_label("Signed SHAP (rating units)",fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    magnitude = pd.Series({f:display[f"family__{f}"].abs().mean() for f in families}).sort_values()
    ax_importance.barh([family_labels[f] for f in magnitude.index],magnitude.values,color="#397b98")
    ax_importance.set_xlabel("Mean |family SHAP| (rating units)",fontsize=6)
    ax_importance.tick_params(axis="y",labelsize=5.5)
    ax_importance.set_title("d  Global attribution magnitude",loc="left",fontsize=7)
    ax_importance.spines[["top","right"]].set_visible(False)
    fig.suptitle("Participant-held-out profiles and model contributions",fontsize=8,y=.99)
    fig.text(.5,.01,"Black dots mark stated conditions; grey dots mark complements. Cut-points are descriptive; contributions are not causal effects.",ha="center",fontsize=5.2)
    # Also provide each component as an independent panel for later layout.
    ax_mean.tick_params(axis="x",labelbottom=True)
    ax_mean.set_xlabel("Profile / participants (n)",fontsize=5.5)
    save_panel(fig,[ax_mean],figures,"f4a1_profile_means")
    ax_mean.tick_params(axis="x",labelbottom=False)
    ax_mean.set_xlabel("")
    save_panel(fig,[ax_signed,cb.ax],figures,"f4a2_signed_contributions")
    save_panel(fig,[ax_condition],figures,"f4a3_conditions")
    save_panel(fig,[ax_importance],figures,"f4a4_global_magnitude")
    savefig(fig,figures,"f4a_condition_profiles")

    country_rows = []
    for country, block in display.groupby("country"):
        for fam in families:
            country_rows.append(dict(country=country,family=fam,n_participants=block.participant_id.nunique(),n_events=len(block),
                mean_signed_contribution=block[f"family__{fam}"].mean(),mean_absolute_contribution=block[f"family__{fam}"].abs().mean(),
                interpretation="Participant-held-out models; known-country setting; small-country descriptives do not establish population precision"))
    bycountry = pd.DataFrame(country_rows)
    bycountry.to_csv(tables/"m2_country_attributions.csv",index=False)
    ccounts = display.groupby("country").participant_id.nunique().sort_values(ascending=True)
    magnitude = bycountry.pivot(index="country",columns="family",values="mean_absolute_contribution").reindex(ccounts.index)
    fig,ax=plt.subplots(figsize=(7.08,8.3))
    left = np.zeros(len(magnitude))
    colors = plt.get_cmap("tab10")(np.linspace(0,.8,len(families)))
    for fam,color in zip(families,colors):
        ax.barh(range(len(magnitude)),magnitude[fam],left=left,label=family_labels[fam],color=color,height=.72)
        left += magnitude[fam].to_numpy()
    short_country={"United Kingdom of Great Britain and Northern Ireland":"United Kingdom","United States of America":"United States"}
    ax.set_yticks(range(len(magnitude)),[f"{short_country.get(c,c)}  (n = {ccounts[c]})" for c in magnitude.index],fontsize=6.5)
    ax.set_xlabel("Mean absolute family contribution (risk-rating units)")
    ax.set_title("Attribution magnitude by reported country; every country retained",fontsize=7)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.07),ncol=2,fontsize=6,frameon=False)
    fig.tight_layout();savefig(fig,figures,"f4b_country_attribution_magnitude")
    write_json(tables.parent/"reports"/"m2_profiles_note.json",dict(
        inspiration="SafeTraffic Copilot, Fig. 6: condition combinations and contribution summaries; no visual or numeric reproduction",
        condition_rules={condition_cols[0]:"trust_pre_acc <= 5",condition_cols[1]:"trust_pre_lks <= 5",condition_cols[2]:"driving_exp_years <= 10"},
        threshold_status="Exploratory descriptive display choices, not validated or confirmatory cut-points",
        eligible_participants=int(profiles.participant_id.nunique()),eligible_events=len(profiles),excluded_missing_condition_events=int((~eligible).sum()),
        profile_ci="1000 participant-cluster draws of fixed held-out predictions/observations, conditional on fitted models and sampled events; no CI if fewer than 5 participants",
        country_precision="No country exclusion. Point descriptions do not quantify population precision, especially for single-participant countries.",
        signed_contributions="Family SHAP sums remain additive with the fold-specific base value",
        magnitude="Mean absolute family SHAP sum; magnitudes are not signed risk differences or causal effects"))


def plot_results(data, out, tables, figures):
    imp = pd.read_csv(tables / "m2_shap_global_importance.csv")
    sv = pd.read_parquet(out / "m2_oof_shap.parquet")
    pairs = pd.read_parquet(out / "m2_sampled_true_interactions.parquet")
    sample = sv.sample(min(6000,len(sv)),random_state=SEED)
    personal = imp[~imp.feature.isin(IDENTITY+["country"])].head(14).iloc[::-1]
    fig,ax=plt.subplots(figsize=(7.4,5.5))
    ax.barh([DISPLAY[c] for c in personal.feature],personal.mean_abs_shap,color="#397b98")
    ax.set_xlabel("Mean absolute held-out SHAP contribution (risk-rating units)")
    ax.set_title("Personal attributes, conditional on questionnaire identity and country",fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout();savefig(fig,figures,"f3a_person_global_attribution")
    numeric_order = imp[imp.feature.isin(NUMERIC)].feature.tolist()
    shap.summary_plot(sample[[f"shap__{c}" for c in numeric_order]].to_numpy(),
        sample[[f"value__{c}" for c in numeric_order]].to_numpy(),
        feature_names=[DISPLAY[c] for c in numeric_order], show=False,plot_size=(8,5.3),max_display=8)
    fig=plt.gcf();plt.xlabel("Held-out SHAP contribution (risk-rating units)")
    plt.title("Continuous and rating-scale pre-task attributes",fontsize=12)
    fig.tight_layout();savefig(fig,figures,"f3b_classic_beeswarm")
    selected=["age","driving_exp_years","trust_pre_acc","trust_pre_lks"]
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,c in zip(axes.ravel(),selected):
        x,y=sample[f"value__{c}"],sample[f"shap__{c}"]
        ax.scatter(x,y,s=5,alpha=.13,color="#397b98",rasterized=True,linewidths=0)
        ax.axhline(0,color="#555555",linewidth=.6)
        ax.set_xlabel(DISPLAY[c]);ax.set_ylabel("Held-out SHAP contribution")
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Model dependence, not an intervention effect",fontsize=13)
    fig.tight_layout();savefig(fig,figures,"f3c_continuous_dependence")
    pairmean = pairs.groupby(["feature_a","feature_b"]).pair_contribution.apply(lambda x:np.abs(x).mean()).reset_index(name="mean_abs")
    matrix=pd.DataFrame(0.,index=NUMERIC,columns=NUMERIC)
    for _,r in pairmean.iterrows():matrix.loc[r.feature_a,r.feature_b]=matrix.loc[r.feature_b,r.feature_a]=r.mean_abs
    fig,ax=plt.subplots(figsize=(8,6.8))
    img=ax.imshow(matrix,cmap="magma_r")
    ax.set_xticks(range(len(NUMERIC)),[DISPLAY[c] for c in NUMERIC],rotation=50,ha="right",fontsize=8)
    ax.set_yticks(range(len(NUMERIC)),[DISPLAY[c] for c in NUMERIC],fontsize=8)
    cb=fig.colorbar(img,ax=ax,shrink=.75);cb.set_label("Mean |2 × TreeSHAP interaction| (rating units)")
    ax.set_title("True pair contributions on a participant-held-out sample",fontsize=11)
    fig.tight_layout();savefig(fig,figures,"f3d_true_interaction_heatmap")
    # Age/experience pair plus strongest remaining continuous pairs;
    # top-pair selection is exploratory and receives no confirmatory p-value.
    strongest=pairmean.sort_values("mean_abs",ascending=False)
    choices=[("age","driving_exp_years")]
    for _,r in strongest.iterrows():
        pair=(r.feature_a,r.feature_b)
        if pair not in choices:choices.append(pair)
        if len(choices)==3:break
    fig,axes=plt.subplots(1,3,figsize=(13.5,4))
    for ax,(a,b) in zip(axes,choices):
        d=pairs[(pairs.feature_a==a)&(pairs.feature_b==b)].groupby("participant_id",as_index=False).agg(
            value_a=("value_a","first"),value_b=("value_b","first"),pair_contribution=("pair_contribution","mean"))
        lim=max(np.abs(d.pair_contribution).quantile(.99),1e-5)
        xx=d.value_a.to_numpy();yy=d.value_b.to_numpy()
        rng=np.random.default_rng(SEED)
        if d.value_a.nunique()<15:xx=xx+rng.uniform(-.1,.1,len(xx))
        if d.value_b.nunique()<15:yy=yy+rng.uniform(-.1,.1,len(yy))
        im=ax.scatter(xx,yy,c=d.pair_contribution,cmap="RdBu_r",norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),s=18,alpha=.8,linewidths=0)
        ax.set_xlabel(DISPLAY[a],fontsize=9);ax.set_ylabel(DISPLAY[b],fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        cb=fig.colorbar(im,ax=ax,shrink=.75);cb.set_label("Pair contribution (2Φᵢⱼ)",fontsize=8)
    fig.suptitle("Mean true pair contribution across sampled events for each held-out participant",fontsize=12)
    fig.tight_layout();savefig(fig,figures,"f3e_pair_contribution_maps")
    metrics=pd.read_csv(tables/"m2_performance.csv")
    protocols=["participant_5fold","leave_country_out","event_group_5fold","crossed_participant_event_5x5"]
    titles=["New participants","New country","New stimuli","New participants + stimuli"]
    fig,axes=plt.subplots(1,4,figsize=(13.8,4.1),sharey=True)
    order=["Training event/block mean","Ridge","LightGBM"]
    for ax,protocol,title in zip(axes,protocols,titles):
        d=metrics[metrics.protocol==protocol].set_index("model").loc[order]
        ax.bar(range(3),d.r2,color=["#a4adb8","#397b98","#df973b"])
        ax.vlines(range(3),d.r2_ci_low,d.r2_ci_high,color="black",lw=.8)
        ax.set_xticks(range(3),["ID reference","Ridge","LightGBM"],rotation=30,ha="right",fontsize=8)
        ax.set_title(title,fontsize=10);ax.axhline(0,color="#555555",lw=.7)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Held-out R²")
    fig.suptitle("All countries retained; performance conditional on operated video ratings",fontsize=12)
    fig.tight_layout();savefig(fig,figures,"f6_generalization")
    plot_profiles(data,out,tables,figures)


def validate(data,out,tables,reports):
    predictions=pd.read_parquet(out/"m2_oof_predictions.parquet")
    source=data.set_index("row_key").risk_event_mean
    assert np.allclose(predictions.actual,predictions.row_key.map(source))
    assert predictions.groupby(["protocol","model"]).size().eq(len(data)).all()
    sv=pd.read_parquet(out/"m2_oof_shap.parquet")
    check=sv.merge(predictions[(predictions.protocol=="participant_5fold")&(predictions.model=="LightGBM")],on="row_key",validate="1:1",suffixes=("_shap","_prediction"))
    assert len(check)==len(data)
    assert np.allclose(check.pred_shap,check.pred_prediction)
    assert np.array_equal(check.participant_id_shap,check.participant_id_prediction)
    cols=[c for c in sv if c.startswith("shap__")]
    error=float(np.abs(sv[cols].sum(axis=1)+sv.base_value-sv.pred).max())
    assert error<1e-8
    interactions=pd.read_parquet(out/"m2_sampled_true_interactions.parquet")
    assert not interactions.duplicated(["row_key","feature_a","feature_b"]).any()
    assert set(interactions.row_key)<=set(data.row_key)
    source_pid=data.set_index("row_key").participant_id
    assert np.array_equal(interactions.participant_id,interactions.row_key.map(source_pid))
    audit=pd.read_csv(tables/"m2_split_audit.csv")
    assert len(audit)==5+data.country.nunique()+5+25 and audit.row_overlap.eq(0).all()
    assert audit[audit.protocol.isin(["participant_5fold","leave_country_out","crossed_participant_event_5x5"])].participant_overlap.eq(0).all()
    assert audit[audit.protocol.isin(["event_group_5fold","crossed_participant_event_5x5"])].event_overlap.eq(0).all()
    assert audit[audit.protocol.eq("leave_country_out")].country_overlap.eq(0).all()
    write_json(reports/"m2_validation.json",dict(source_outcomes_match=True,each_protocol_covers_rows_once=True,
        n_prediction_rows=len(predictions),n_protocols=predictions.protocol.nunique(),n_models=predictions.model.nunique(),n_outer_folds=len(audit),
        split_intersections_pass=True,
        oof_shap_rows=len(sv),shap_keys_and_predictions_match=True,max_shap_additivity_error=error,
        interaction_rows=interactions.row_key.nunique(),interaction_participants=interactions.participant_id.nunique(),
        interaction_keys_match_source=True,
        positive_operated_outcome_only=True,prepost_zeros_retained=True,post_outcome_features_included=False,
        physical_mapping_used=False,interaction_validation="m2_interaction_validation.json"))


def results_summary(participants,data,tables,reports):
    attribution=pd.read_csv(tables/"m2_shap_global_importance.csv").set_index("feature")
    sensitivity=pd.read_csv(tables/"m2_age_experience_sensitivity_attribution.csv")
    changes=[]
    for omitted,retained in [("age","driving_exp_years"),("driving_exp_years","age")]:
        block=sensitivity[(sensitivity.specification==f"without_{omitted}")&(sensitivity.feature==retained)]
        changes.append(dict(omitted=omitted,retained=retained,
            primary_mean_abs_shap=attribution.loc[retained,"mean_abs_shap"],
            sensitivity_mean_abs_shap=float(np.average(block.mean_abs_shap,weights=block.n))))
    preitems=[c for c in NUMERIC if c.startswith(("trust_pre","accept_pre"))]
    correlation=participants[["age","driving_exp_years"]]
    write_json(reports/"m2_results_summary.json",dict(
        primary_sample=dict(participants=int(data.participant_id.nunique()),countries=int(data.country.nunique()),observed_events=len(data)),
        model_scope="Verified questionnaire identity plus 24 pre-task attributes; no kinematic predictors or post-task attributes",
        zero_preitems_retained=participants[preitems].eq(0).sum().to_dict(),
        matched_algorithm_increments=pd.read_csv(tables/"m2_person_increment.csv").to_dict(orient="records"),
        performance=pd.read_csv(tables/"m2_performance.csv").to_dict(orient="records"),
        age_experience_pearson=float(correlation.corr().iloc[0,1]),
        age_experience_spearman=float(correlation.corr(method="spearman").iloc[0,1]),
        correlated_covariate_attribution_sensitivity=changes,
        interpretation=[
            "The matched-model personal-attribute increments for new participants and new countries have participant-bootstrap intervals spanning zero.",
            "Performance is conditional on operated video ratings, the selected model class/candidates, and this questionnaire stimulus set.",
            "Event-only holdout retains the same people across training and evaluation; its performance must not be read as generalization to new people.",
            "SHAP explains fitted predictions. Correlated attributes can substitute for one another, and model attributions are not independent psychological contributions or causal effects.",
            "Metric confidence intervals condition on fitted OOF models and resample participants, not countries or events. No claim of small-country population precision is made.",
            "No physical mapping sensitivity was rerun in round 2; unresolved video-to-kinematics links preclude physical mechanism claims."
        ]))
    descriptions={
        "f3a_person_global_attribution":"Mean absolute source-variable SHAP; personal variables only, conditional on stimulus identity and country",
        "f3b_classic_beeswarm":"SHAP distribution for 8 continuous/rating-scale pre-task attributes; display sample 6000 event rows",
        "f3c_continuous_dependence":"Raw age, experience and initial ACC/LKS trust versus held-out SHAP; display sample 6000 event rows",
        "f3d_true_interaction_heatmap":"Mean absolute true unordered TreeSHAP pair contribution (2*Phi_ij); 400 events from 100 held-out participants",
        "f3e_pair_contribution_maps":"True pair contribution averaged over sampled events per held-out participant; age/experience plus two exploratory largest remaining numeric pairs",
        "f4a_condition_profiles":"Four-panel review composition of descriptive profiles, signed contributions, condition matrix and magnitude",
        "f4a1_profile_means":"Separate panel: fixed-OOF predicted and observed profile means with 1000 participant-bootstrap confidence intervals",
        "f4a2_signed_contributions":"Separate panel: mean signed family SHAP by descriptive profile",
        "f4a3_conditions":"Separate panel: 8 descriptive condition combinations and participant counts",
        "f4a4_global_magnitude":"Separate panel: mean absolute summed family SHAP, all OOF model rows",
        "f4b_country_attribution_magnitude":"All 29 countries; participant-held-out known-stimulus attribution magnitudes and sample counts; no population precision claim",
        "f6_generalization":"Held-out R2 across 4 targets, training event/block mean reference and full Ridge/LightGBM, conditional participant-bootstrap metric intervals"
    }
    records=[]
    for stem,description in descriptions.items():
        for suffix in ["png","pdf"]:
            path=tables.parent/"figures"/f"{stem}.{suffix}"
            assert path.exists()
            records.append(dict(file=path.name,format=suffix,description=description,bytes=path.stat().st_size,sha256=stable_digest(path)))
    pd.DataFrame(records).to_csv(tables/"m2_figure_metadata.csv",index=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage",choices=["all","fit","comparators","crossed","explain","plot"],default="all")
    ap.add_argument("--data-dir",type=Path,default=DATA_DIR)
    ap.add_argument("--output-dir",type=Path,default=ROOT/"outputs/core")
    ap.add_argument("--interaction-participants-per-fold",type=int,default=20)
    args=ap.parse_args();started=time.monotonic()
    out,tables,reports,figures=[args.output_dir/p for p in ["prediction","tables","reports","figures"]]
    for p in [out,tables,reports,figures]:p.mkdir(parents=True,exist_ok=True)
    participants,data,coverage,manifest=load_data(args.data_dir)
    coverage.to_csv(tables/"m2_observation_coverage_country.csv",index=False)
    data[["row_key","participant_id","scenario","event_key","country","n_valid_windows","risk_event_mean"]].to_parquet(out/"m2_analysis_sample.parquet",index=False)
    with threadpool_limits(limits=2):
        if args.stage in ["all","fit"]:fit_all(data,out,tables,reports)
        if args.stage == "comparators":fit_identity_comparators(data,out,tables,reports)
        if args.stage == "crossed":refit_crossed(data,out,tables,reports)
        if args.stage in ["all","explain"]:explain(data,out,tables,reports,args.interaction_participants_per_fold)
        if args.stage in ["all","comparators","crossed","explain","plot"]:
            plot_results(data,out,tables,figures)
            validate(data,out,tables,reports)
            results_summary(participants,data,tables,reports)
    manifest.update(stage=args.stage,seed=SEED,outer_cv="participant5; all-country LOCO; event5; crossed participant5/event5",
        tuning="Ridge 3 alphas and LightGBM 2 fixed candidates; inner grouped 3-fold CV for participant/country/event targets, inner participant3 x event3 crossed CV for the joint target",
        tree_candidates=GBM_GRID,test_early_stopping=False,shap_version=shap.__version__,lightgbm_version=lgb.__version__,
        prediction_interval_note="Metric CIs resample participants from fixed OOF predictions; no individual prediction intervals, retraining, country or event sampling",
        interaction_sample_participants_per_fold=args.interaction_participants_per_fold,
        script_sha256=stable_digest(Path(__file__)),helper_provenance="Generic utilities inlined; no dependency on earlier analysis scripts",
        elapsed_seconds=time.monotonic()-started)
    write_json(reports/"m2_manifest.json",manifest)
    print(f"DONE {args.stage} {time.monotonic()-started:.1f}s",flush=True)


if __name__=="__main__":main()
