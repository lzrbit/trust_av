"""Observed risk-score distributions, with non-operation separated from scores."""
from settings import WORKSPACE as ROOT, DATA_DIR, QUESTIONNAIRE, REPO, relative_label
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


OUT=ROOT/'outputs/core'

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    for folder in ['tables','figures','reports']:(OUT/folder).mkdir(parents=True,exist_ok=True)
    source=DATA_DIR / 'windows.parquet';before=digest(source);w=pd.read_parquet(source)
    valid=w.risk.dropna();assert valid.between(1,10).all() and np.allclose(valid,valid.round())
    assert w.loc[w.risk_raw.eq(0),'risk'].isna().all()
    nvalid=len(valid);nzero=int(w.risk_raw.eq(0).sum());npeople=w.participant_id.nunique()
    rows=[]
    for group,d in [('All',w)]+list(w.groupby('scenario')):
        v=d.risk.dropna();counts=v.value_counts().reindex(range(1,11),fill_value=0)
        for score,count in counts.items():rows.append({'scenario':group,'risk_score':score,'n_ratings':int(count),'proportion_valid_ratings':count/len(v),'n_valid_ratings':len(v),'n_participants':d.participant_id.nunique(),'n_zero_no_operation':int(d.risk_raw.eq(0).sum())})
    table=pd.DataFrame(rows);table.to_csv(OUT/'tables/d2_risk_score_distribution.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    def save(fig,stem):
        for suffix in ['pdf','svg','png']:fig.savefig(OUT/f'figures/{stem}.{suffix}',bbox_inches='tight',dpi=220)
        plt.close(fig)
    a=table[table.scenario.eq('All')];fig,ax=plt.subplots(figsize=(6.4,4.0))
    ax.bar(a.risk_score,100*a.proportion_valid_ratings,color='#268E8D',width=.76)
    for r in a.itertuples():ax.text(r.risk_score,100*r.proportion_valid_ratings+.6,f'{r.n_ratings:,}',ha='center',va='bottom',fontsize=8)
    ax.set(xticks=range(1,11),xlabel='Risk score when operated (1–10)',ylabel='Percentage of valid ratings',ylim=(0,100*a.proportion_valid_ratings.max()*1.2),title='Distribution of observed risk ratings')
    ax.text(.98,.97,f'{nvalid:,} repeated ratings\n{npeople:,} participants',transform=ax.transAxes,ha='right',va='top',fontsize=9)
    ax.text(.0,-.25,f'{nzero:,} zeros indicate no operation and are excluded from score bars.',transform=ax.transAxes,fontsize=8,color='#A25448')
    save(fig,'f1_l_risk_distribution')
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.4),sharex=True,sharey=True)
    ymax=100*table[table.scenario.ne('All')].proportion_valid_ratings.max()*1.2
    colors=['#2563A6','#D17A22','#16857D','#8B4B9E']
    for ax,block,color in zip(axes.ravel(),['B1','B2','B3','B4'],colors):
        q=table[table.scenario.eq(block)];ax.bar(q.risk_score,100*q.proportion_valid_ratings,color=color,width=.76)
        ax.set(xticks=range(1,11),ylim=(0,ymax),title=block)
        ax.text(.98,.96,f'{q.n_valid_ratings.iloc[0]:,} valid\n{q.n_zero_no_operation.iloc[0]:,} no operation',transform=ax.transAxes,ha='right',va='top',fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('Operated risk score')
    for ax in axes[:,0]:ax.set_ylabel('Valid ratings (%)')
    fig.suptitle('Risk-score distributions by questionnaire block',fontsize=12)
    fig.text(.5,.01,'Each participant provides repeated ratings. Zero denotes no operation and is omitted.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.035,1,.96]);save(fig,'f1_m_risk_distribution_by_block')
    assert digest(source)==before
    (OUT/'reports/d2_risk_distribution.json').write_text(json.dumps({'n_participants':int(npeople),'n_valid_repeated_ratings':nvalid,'n_zero_no_operation':nzero,'unit':'rating-level descriptive proportions; observations are repeated within people, not independent replicates','zero_rule':'video risk 0 is no operation and omitted from score distribution; shown as a separate count','input_sha256':before,'script_sha256':digest(Path(__file__)),'figure_stems':['f1_l_risk_distribution','f1_m_risk_distribution_by_block']},indent=2))
    print(f'{nvalid:,} positive risk ratings; {nzero:,} no-operation zeros; {npeople:,} participants')

if __name__=='__main__':main()
