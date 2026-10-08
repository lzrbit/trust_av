"""Joint robust contrasts of risk coefficients across trust outcomes.

All regressions use the same participant rows. Cross-outcome covariance is
computed from their joint estimating equations; separate P values are never
used as a test of a difference between outcomes.
"""
from settings import WORKSPACE as ROOT, DATA_DIR, QUESTIONNAIRE, REPO, relative_label
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=ROOT/'outputs/core'

def main():
    p=pd.read_parquet(DATA_DIR / 'participants.parquet')
    f=pd.read_csv(OUT/'tables/a2_participant_risk_features.csv')
    d=p.merge(f,on='participant_id',validate='1:1')
    features=['mean_risk','excursion','late_change','positive_fraction']
    for c in features:d[c+'_z']=(d[c]-d[c].mean())/d[c].std(ddof=1)
    d['age_10']=(d.age-d.age.mean())/10;d['experience_10']=(d.driving_exp_years-d.driving_exp_years.mean())/10
    d['gender']=d.gender.fillna('Missing').astype(str);d['country']=d.country.fillna('Unknown').astype(str)
    required=[f'{x}_z' for x in features]+['age_10','experience_10']+[f'trust_{t}_{item}' for t in ['pre','post'] for item in ['overall','acc','lks']]
    d=d.dropna(subset=required).copy()
    models={}
    for item in ['overall','acc','lks']:
        formula=f'trust_post_{item} ~ trust_pre_{item} + mean_risk_z + excursion_z + late_change_z + positive_fraction_z + age_10 + experience_10 + C(gender) + C(country)'
        models[item]=smf.ols(formula,data=d).fit(cov_type='HC1')
    rows=[];maxerr=0
    for first,second in [('lks','acc'),('lks','overall'),('acc','overall')]:
        a,b=models[first],models[second];xa=a.model.exog;xb=b.model.exog;n=len(d)
        assert np.array_equal(a.model.data.row_labels,b.model.data.row_labels)
        pa=np.linalg.pinv(xa.T@xa);pb=np.linalg.pinv(xb.T@xb)
        fa=n/a.df_resid;fb=n/b.df_resid
        ca=pa@(xa.T@((np.asarray(a.resid)**2)[:,None]*xa))@pa*fa
        maxerr=max(maxerr,float(np.max(np.abs(ca-np.asarray(a.cov_params())))))
        cross=pa@(xa.T@((np.asarray(a.resid)*np.asarray(b.resid))[:,None]*xb))@pb*np.sqrt(fa*fb)
        for feature in features[:3]:
            term=feature+'_z';ia=a.model.exog_names.index(term);ib=b.model.exog_names.index(term)
            delta=a.params[term]-b.params[term]
            variance=a.cov_params().loc[term,term]+b.cov_params().loc[term,term]-2*cross[ia,ib]
            assert variance>0
            se=np.sqrt(variance);z=delta/se
            rows.append(dict(first=first,second=second,contrast=f'{first} minus {second}',feature=feature,n=n,estimate=delta,se=se,ci_low=delta-1.96*se,ci_high=delta+1.96*se,p=2*stats.norm.sf(abs(z))))
    result=pd.DataFrame(rows);result['q']=multipletests(result.p,method='fdr_bh')[1]
    assert maxerr<1e-8
    result.to_csv(OUT/'tables/a2_trust_between_item_coefficients.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(6.4,3.8))
    for k,(contrast,color) in enumerate(zip(result.contrast.unique(),['#2563A6','#16857D','#D17A22'])):
        s=result[result.contrast==contrast].set_index('feature').reindex(features[:3]);y=np.arange(3)+(k-1)*.18
        ax.errorbar(s.estimate,y,xerr=np.vstack([s.estimate-s.ci_low,s.ci_high-s.estimate]),fmt='o',ms=4,capsize=2,label=contrast.upper(),color=color)
    ax.set_yticks(range(3),['Mean risk','Excursion','Late change']);ax.invert_yaxis();ax.axvline(0,color='black',lw=.7)
    ax.set(xlabel='Difference in adjusted post-trust coefficient per risk SD',title='Direct comparison of associations across trust items')
    ax.legend(frameon=False,fontsize=8);fig.tight_layout()
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'figures/f5_between_item_association_contrasts.{ext}',bbox_inches='tight',dpi=200)
    plt.close(fig)
    (OUT/'reports/a2_trust_coefficient_contrasts.json').write_text(json.dumps({'status':'passed','n_participants':len(d),'all_countries_in_complete_summary_sample':int(d.country.nunique()),'covariance_check_max_error':maxerr,'method':'Joint heteroskedasticity-robust cross-outcome sandwich; model-specific pre-task trust; 9 contrasts BH-adjusted','scope':'Associational; outcome-specific baseline adjustment; normal intervals conditional on feature definition'},indent=2))
    print(result.to_string(index=False))

if __name__=='__main__':main()
