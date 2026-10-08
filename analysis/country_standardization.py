"""Descriptive country differences after adjusting observed event composition.

For each operated rating, subtract the same event-position mean estimated in
other countries. Average within participant before country summaries. This is
standardization to the observed stimuli, not a causal country effect.
"""
from settings import WORKSPACE as ROOT, DATA_DIR, QUESTIONNAIRE, REPO, relative_label
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=ROOT/'outputs/core'

def short(c):
    return {'United Kingdom of Great Britain and Northern Ireland':'United Kingdom','United States of America':'United States'}.get(c,c)

def main():
    p=pd.read_parquet(DATA_DIR / 'participants.parquet').set_index('participant_id',drop=False)
    w=pd.read_parquet(DATA_DIR / 'windows.parquet').dropna(subset=['risk']).copy()
    key=['event_key','window_index']
    total=w.groupby(key).risk.agg(total_sum='sum',total_n='size').reset_index()
    own=w.groupby(['country',*key]).risk.agg(own_sum='sum',own_n='size').reset_index()
    v=w.merge(total,on=key,validate='m:1').merge(own,on=['country',*key],validate='m:1')
    assert (v.total_n-v.own_n).gt(0).all()
    v['other_country_event_position_mean']=(v.total_sum-v.own_sum)/(v.total_n-v.own_n)
    v['residual_from_other_countries']=v.risk-v.other_country_event_position_mean
    participant=v.groupby(['participant_id','scenario','window_index']).residual_from_other_countries.mean().unstack(['scenario','window_index'])
    participant=participant.reindex(p.index)
    participant.columns=[f'{b}_w{k}' for b,k in participant.columns]
    individual=participant.mean(axis=1)
    rows=[]; rng=np.random.default_rng(20261014)
    for country,ids in p.groupby('country').groups.items():
        a=individual.loc[ids].dropna().to_numpy();n=len(a);lo=hi=np.nan
        if n>1:
            weights=rng.multinomial(n,np.full(n,1/n),size=2000)
            lo,hi=np.quantile(weights@a/n,[.025,.975])
        rows.append(dict(country=country,n_participants=n,stimulus_adjusted_difference=a.mean(),ci_low=lo,ci_high=hi,interpretation='participant-weighted difference from other-country ratings of the same event-position; external estimates held fixed'))
    table=pd.DataFrame(rows).sort_values('n_participants',ascending=False)
    table.to_csv(OUT/'tables/a2_country_stimulus_standardized.csv',index=False)
    profiles=participant.groupby(p.country).mean().reindex(table.country)
    profiles.to_csv(OUT/'tables/a2_country_stimulus_residual_profiles.csv',index_label='country')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(6.3,8.5))
    for i,(_,r) in enumerate(table.iterrows()):
        if np.isfinite(r.ci_low):ax.plot([r.ci_low,r.ci_high],[i,i],color='#777777',lw=.9)
        ax.scatter(r.stimulus_adjusted_difference,i,s=20,color='#2563A6' if r.n_participants>=30 else '#999999')
    ax.set_yticks(range(len(table)),[f'{short(r.country)} (n={r.n_participants})' for _,r in table.iterrows()],fontsize=8)
    ax.invert_yaxis();ax.axvline(0,color='black',lw=.7)
    ax.set(xlabel='Difference from other-country ratings of matched stimuli',title='Country differences after stimulus matching')
    fig.tight_layout()
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'figures/f2_country_stimulus_standardized.{ext}',bbox_inches='tight',dpi=200)
    plt.close(fig)
    metadata={'status':'passed','countries':len(table),'ratings':len(v),'participants':len(p),
      'external_reference':'Every country is excluded when estimating its event-position reference','summary':'equal observed positions within participant, then equal participants within country',
      'ci':'2000 participant-bootstrap percentile; external matched-stimulus mean fixed; one-participant countries have no estimable within-country bootstrap CI',
      'not_adjusted':['demographic composition','response selection','unobserved sampling differences'],
      'not_supported':'causal cultural effect or representative national estimate'}
    (OUT/'reports/a2_country_standardization.json').write_text(json.dumps(metadata,indent=2))
    assert len(table)==p.country.nunique()
    print(table[['country','n_participants','stimulus_adjusted_difference']].to_string(index=False))

if __name__=='__main__':main()
