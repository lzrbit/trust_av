"""Harmonize six paired pre/post item summaries to the existing 2,000-draw CI.

Uses the existing analyse_dynamics.bootstrap_means implementation and SEED.
Pre, post and post-minus-pre are evaluated on the same finite paired records;
zeros remain valid observations. No models are fitted or source data changed.
"""
from settings import WORKSPACE as ROOT, DATA_DIR, QUESTIONNAIRE, REPO, relative_label
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd




def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    root = parser.parse_args().project_root.resolve()
    source = DATA_DIR / 'participants.parquet'
    old = root/'outputs/core/tables/a2_trust_changes.csv'
    code = REPO/'analysis/analyse_dynamics.py'
    out = root/'outputs/prepost'
    for folder in ('tables','reports'): (out/folder).mkdir(parents=True,exist_ok=True)
    spec = importlib.util.spec_from_file_location('existing_dynamics',code)
    dynamics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dynamics)
    assert dynamics.SEED == 20261012 and dynamics.B == 2000
    p = pd.read_parquet(source)
    assert len(p)==2164 and p.participant_id.is_unique
    previous = pd.read_csv(old,float_precision='round_trip').set_index('item')
    rows, checks = [], []
    settings = [('trust','overall','Overall trust'), ('trust','acc','ACC trust'),
                ('trust','lks','LKS trust'), ('accept','delegate','Delegate driving'),
                ('accept','monitor_raw','Need for supervision'),
                ('accept','distract','Non-driving activities')]
    for family,item,label in settings:
        pre_col, post_col = f'{family}_pre_{item}', f'{family}_post_{item}'
        pre,post = p[pre_col].to_numpy(float),p[post_col].to_numpy(float)
        valid = np.isfinite(pre)&np.isfinite(post)
        pre,post = pre[valid],post[valid]
        assert len(pre)==2164 and np.all((pre>=0)&(pre<=10)) and np.all((post>=0)&(post<=10))
        row = dict(family=family,item=item,label=label,pre_col=pre_col,post_col=post_col,
                   n=int(valid.sum()),n_pre_zero=int((pre==0).sum()),n_post_zero=int((post==0).sum()),
                   zero_is_valid=True,bootstrap_draws=dynamics.B,seed=dynamics.SEED)
        for metric,values in [('pre',pre),('post',post),('delta',post-pre)]:
            ci,_ = dynamics.bootstrap_means(values)
            row.update({f'{metric}_mean':float(values.mean()),f'{metric}_ci_low':float(ci[0,0]),f'{metric}_ci_high':float(ci[1,0])})
        assert np.isclose(row['delta_mean'],row['post_mean']-row['pre_mean'],atol=1e-14,rtol=0)
        if family=='trust':
            fields=['delta','ci_low','ci_high']
            computed=np.array([row['delta_mean'],row['delta_ci_low'],row['delta_ci_high']])
            saved=previous.loc[item,fields].to_numpy(float)
            difference=np.abs(computed-saved)
            checks.append(dict(item=item,fields=fields,saved=saved.tolist(),computed=computed.tolist(),
                               maximum_absolute_difference=float(difference.max()),
                               exact_float_equal=bool(np.array_equal(computed,saved)),
                               matches_within_csv_float_precision=bool(np.all(difference<1e-14))))
            assert np.all(difference<1e-14),(item,computed,saved)
        rows.append(row)
    table=out/'tables/prepost_harmonized.csv'
    pd.DataFrame(rows).to_csv(table,index=False)
    report=dict(status='pass',participants=2164,item_count=6,bootstrap_draws=dynamics.B,seed=dynamics.SEED,
                resampling_unit='participant',method='Existing multinomial participant-weight bootstrap; 2.5 and 97.5 percentiles',
                pairing='Pre and post summaries and changes use identical finite paired participants for each item',
                trust_acceptance_zero='Valid; retained in means and every bootstrap draw',
                independence_scope='Intervals condition on the observed cohort sampling frame; no model fitting',
                trust_change_reproduction=checks,
                maximum_trust_reproduction_error=max(x['maximum_absolute_difference'] for x in checks),
                source_hashes={relative_label(f,root):sha(f) for f in [source,old,code]},
                output_hashes={str(table.relative_to(root)):sha(table)})
    (out/'reports/prepost_harmonization_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
