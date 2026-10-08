"""Crossed participant/event variance on the new positive-score outcome."""
from settings import WORKSPACE as ROOT, DATA_DIR, QUESTIONNAIRE, REPO, relative_label
from pathlib import Path
import json
import signal
import time
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMParams
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=ROOT/'outputs/core'

def timeout(signum,frame):raise TimeoutError('Crossed model time limit reached')

def main():
    e=pd.read_parquet(DATA_DIR / 'events.parquet')
    e=e[e.risk_event_mean.notna()].copy()
    for c in ['participant_id','event_key','country','gender']:e[c]=e[c].fillna('Missing').astype(str)
    e['one_group']=1;e['age_10']=(e.age-e.age.mean())/10;e['experience_10']=(e.driving_exp_years-e.driving_exp_years.mean())/10
    formula='risk_event_mean ~ C(scenario) + age_10 + experience_10 + trust_pre_overall + trust_pre_acc + trust_pre_lks + C(gender) + C(country)'
    results=[];metadata=[]
    for sample,d in [('all_operated_events',e),('all_positions_operated',e[e.complete_valid_windows])]:
        record={'sample':sample,'n_events':len(d),'n_participants':int(d.participant_id.nunique()),'n_countries':int(d.country.nunique()),'formula':formula,'status':'pending'}
        old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,120)
        start=time.monotonic()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model=smf.mixedlm(formula,d,groups='one_group',re_formula='0',vc_formula={'participant':'0+C(participant_id)','questionnaire_event':'0+C(event_key)'},use_sparse=True)
                start_params=MixedLMParams.from_components(fe_params=smf.ols(formula,d).fit().params.to_numpy(),cov_re=np.zeros((0,0)),vcomp=np.array([1.,.5]))
                fitted=model.fit(reml=True,method='lbfgs',maxiter=100,disp=False,start_params=start_params)
            record['warnings']=[str(x.message) for x in caught];record['log_likelihood']=float(fitted.llf);record['finite_estimates']=bool(np.isfinite(fitted.params).all() and np.isfinite(fitted.llf));record['status']='converged' if fitted.converged and record['finite_estimates'] else 'not_converged'
            if record['status']=='converged':
                for name,value in zip(model.exog_vc.names,fitted.vcomp):results.append(dict(sample=sample,component=name,variance=float(value)))
                results.append(dict(sample=sample,component='residual',variance=float(fitted.scale)))
                ci=fitted.conf_int().loc[fitted.fe_params.index]
                pd.DataFrame({'term':fitted.fe_params.index,'estimate':fitted.fe_params.values,'ci_low':ci.iloc[:,0].values,'ci_high':ci.iloc[:,1].values}).to_csv(OUT/f'tables/a2_mixed_coefficients_{sample}.csv',index=False)
        except Exception as ex:record['status']='failed';record['error']=repr(ex)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        record['elapsed_seconds']=time.monotonic()-start;metadata.append(record);print(json.dumps(record),flush=True)
    table=pd.DataFrame(results);table.to_csv(OUT/'tables/a2_mixed_variance_components.csv',index=False)
    (OUT/'reports/a2_mixed_model_metadata.json').write_text(json.dumps({'models':metadata,'interpretation':'Model-conditional random-intercept variance; country fixed effects; not variance causally attributable to demographics, countries or stimuli','repeated_structure':'Crossed participants and questionnaire events; positive-score event means'},indent=2))
    draw_diagnostic(table)

def draw_diagnostic(table):
    if len(table):
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
        fig,ax=plt.subplots(figsize=(5.8,3.7));items=['participant','questionnaire_event','residual']
        for i,(sample,label,color) in enumerate([('all_operated_events','All usable events','#2563A6'),('all_positions_operated','Complete operated events','#c2c2c2')]):
            s=table[table['sample']==sample].set_index('component').reindex(items)
            if not s.variance.isna().all():ax.bar(np.arange(3)+(i-.5)*.3,s.variance,width=.28,color=color,label=label)
        ax.set(xticks=range(3),xticklabels=['Participant','Questionnaire event','Residual'],ylabel='Conditional variance (rating points squared)',title='Diagnostic: crossed random-intercept model')
        ax.legend(frameon=False,fontsize=8);fig.tight_layout(rect=[0,.11,1,1])
        fig.text(.5,.025,'Numerical warnings during fitting; finite final estimates.\nNot used for the main inference.',ha='center',fontsize=8,color='#8B3A3A')
        for ext in ['pdf','svg','png']:fig.savefig(OUT/f'figures/f2_crossed_variance_components.{ext}',bbox_inches='tight',dpi=200)
        plt.close(fig)

if __name__=='__main__':
    import sys
    if '--plot-only' in sys.argv:
        draw_diagnostic(pd.read_csv(OUT/'tables/a2_mixed_variance_components.csv'))
    else:
        main()
