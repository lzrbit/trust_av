"""Exploratory six-comparison Bonferroni sensitivity for shape increments.

Intervals resample participants from fixed OOF predictions; they do not include
retraining uncertainty. The six-item family was specified after inspecting the
results, so these remain sensitivity analyses rather than confirmatory tests.
"""
from settings import WORKSPACE as ROOT, DATA_DIR, QUESTIONNAIRE, REPO, relative_label
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


OUT = ROOT / 'outputs/core'
SEED = 20261022
DRAWS = 6000

def main():
    source = OUT / 'prediction/a2_trust_oof.parquet'
    d = pd.read_parquet(source)
    ids = np.sort(d.participant_id.unique())
    n = len(ids)
    rng = np.random.default_rng(SEED)
    weights = rng.multinomial(n, np.full(n, 1/n), size=DRAWS)
    rows = []
    for protocol in ['participant_5fold', 'leave_country_out']:
        for item in ['overall', 'acc', 'lks']:
            part = d[(d.protocol == protocol) & (d.item == item)]
            base = part[part.feature_set == 'pre_task_plus_mean_risk_response'].set_index('participant_id').reindex(ids)
            full = part[part.feature_set == 'pre_task_plus_both'].set_index('participant_id').reindex(ids)
            assert base.index.is_unique and full.index.is_unique
            assert np.array_equal(base.actual, full.actual)
            assert np.array_equal(base.fold, full.fold)
            y = base.actual.to_numpy()
            improvement = (y-base.pred.to_numpy())**2 - (y-full.pred.to_numpy())**2
            sst = np.sum((y-y.mean())**2)
            bootstrap_sst = weights@(y*y) - (weights@y)**2/n
            assert np.all(bootstrap_sst > 0)
            delta = (weights@improvement)/bootstrap_sst
            lo, hi = np.quantile(delta, [.05/(2*6), 1-.05/(2*6)])
            rows.append(dict(protocol=protocol,item=item,delta_r2=improvement.sum()/sst,
                             ci_low_bonferroni_6=lo,ci_high_bonferroni_6=hi,
                             bootstrap_draws=DRAWS,
                             family='six outcome/protocol shape-above-mean contrasts',
                             scope='exploratory fixed-OOF participant bootstrap; Bonferroni percentile sensitivity'))
    pd.DataFrame(rows).to_csv(OUT/'tables/a2_shape_increment_multiplicity_sensitivity.csv',index=False)
    (OUT/'reports/a2_multiplicity_sensitivity.json').write_text(json.dumps({
        'seed':SEED,'draws':DRAWS,'family_size':6,'n_participants':n,
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'scope':'Post hoc exploratory sensitivity; fixed OOF participant resampling; no refitting; not country-cluster intervals.',
        'tail_quantiles':[.05/(2*6),1-.05/(2*6)]
    },indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == '__main__':
    main()

