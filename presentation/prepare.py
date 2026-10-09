"""Rebuild display-only dispersion summaries; no model fitting or new intervals."""
import json
import numpy as np
import pandas as pd
from .base import digest

def prepare(builder,items=None):
    items=set(items or ['prepost','positions'])
    written=[]
    if 'prepost' in items:
        p=pd.read_parquet(builder.data/'participants.parquet')
        source=builder.root/'outputs/prepost/tables/prepost_harmonized.csv'
        table=pd.read_csv(source)
        for phase in ['pre','post']:
            values=[]
            for r in table.itertuples():
                x=p[getattr(r,phase+'_col')].dropna().to_numpy(float)
                if len(x)!=r.n or not np.isclose(x.mean(),getattr(r,phase+'_mean'),atol=1e-12,rtol=0):
                    raise ValueError('Pre/post table does not match authorized participant input')
                values.append(float(x.std(ddof=1)))
            table[phase+'_sd']=values
        target=builder.dest/'tables/prepost_dispersion.csv';table.to_csv(target,index=False);written.append(target)
    if 'positions' in items:
        p=pd.read_parquet(builder.data/'participants.parquet')
        w=pd.read_parquet(builder.data/'windows.parquet')
        if (w.risk.dropna()<=0).any():raise ValueError('Derived risk must encode no-operation zeros as missing')
        d=builder.table('a2_position_means')
        profile=w.groupby(['participant_id','scenario','window_index']).risk.mean().unstack(['scenario','window_index']).reindex(p.participant_id)
        rows=[]
        for r in d.itertuples():
            values=profile[(r.scenario,r.position)].dropna().to_numpy(float)
            if len(values)!=r.n_participants or not np.isclose(values.mean(),r.mean,rtol=0,atol=1e-12):
                raise ValueError('Saved position means do not match authorized windows')
            sd=float(values.std(ddof=1));rows.append({'scenario':r.scenario,'position':r.position,'n':len(values),'mean':r.mean,'sd':sd,'ci_low':r.ci_low,'ci_high':r.ci_high,'mean_minus_sd':r.mean-sd,'mean_plus_sd':r.mean+sd})
        target=builder.dest/'tables/participant_position_dispersion.csv';pd.DataFrame(rows).to_csv(target,index=False);written.append(target)
    (builder.reportdir/'display_summary_validation.json').write_text(json.dumps({'status':'pass','operation':'Sample SD (ddof=1); means and intervals checked against saved estimates; no resampling','files':{builder.relative(p):digest(p) for p in written}},indent=2)+'\n')
    return written
