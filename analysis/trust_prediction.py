"""Nested held-out prediction of post-task trust, with train-only risk features.

Every inner and outer training partition selects its own block high-point
positions. Trust/acceptance zero is valid; video risk zero is no operation.
"""
from settings import WORKSPACE as ROOT, DATA_DIR, QUESTIONNAIRE, REPO, relative_label
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GridSearchCV
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


OUT=ROOT/'outputs/core'
SEED=20261013
BLOCKS={'B1':6,'B2':5,'B3':5,'B4':5}
PROFILE=[f'{block}_w{k}' for block,n in BLOCKS.items() for k in range(1,n+1)]
BASE=['age','driving_exp_years','trust_pre_overall','trust_pre_acc','trust_pre_lks',
      'accept_pre_delegate','accept_pre_monitor_raw','accept_pre_distract']
CATEGORIES=['country','gender','education']
DYNAMICS=['mean_risk','excursion','late_change']
FEATURES={'pre_task':[], 'pre_task_plus_response':['positive_fraction'],
          'pre_task_plus_dynamics':DYNAMICS, 'pre_task_plus_both':DYNAMICS+['positive_fraction'],
          'pre_task_plus_mean_risk_response':['mean_risk','positive_fraction']}
COMPARISONS=[('pre_task','pre_task_plus_response'),('pre_task','pre_task_plus_dynamics'),
             ('pre_task','pre_task_plus_both'),('pre_task_plus_response','pre_task_plus_both'),
             ('pre_task_plus_dynamics','pre_task_plus_both'),
             ('pre_task','pre_task_plus_mean_risk_response'),
             ('pre_task_plus_mean_risk_response','pre_task_plus_both')]


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


class TrainingRiskSummaries(BaseEstimator,TransformerMixin):
    """Learn high-point positions from X_train only, then apply unchanged.

    mean_risk gives equal weight to the 21 observed profile positions. Each
    excursion/late change is a mean of available paired block contrasts.
    """
    def __init__(self,include_dynamics=True,include_mean=True):
        self.include_dynamics=include_dynamics;self.include_mean=include_mean
    def fit(self,X,y=None):
        self.peaks_={}
        if self.include_dynamics:
            for block,n in BLOCKS.items():
                cols=[f'{block}_w{k}' for k in range(1,n+1)]
                means=X[cols].mean()
                if not means.notna().any():raise ValueError(f'No observed training profile in {block}')
                self.peaks_[block]=int(means.idxmax().rsplit('w',1)[1])
        return self
    def transform(self,X):
        d=X.copy()
        if self.include_mean:d['mean_risk']=d[PROFILE].mean(axis=1)
        if self.include_dynamics:
            if len(self.peaks_)!=4:raise ValueError('Risk feature transformer has not been fitted')
            ex=[];late=[]
            for block,n in BLOCKS.items():
                peak=d[f'{block}_w{self.peaks_[block]}']
                ex.append(peak-d[f'{block}_w1'])
                late.append(d[f'{block}_w{n}']-peak)
            d['excursion']=pd.concat(ex,axis=1).mean(axis=1)
            d['late_change']=pd.concat(late,axis=1).mean(axis=1)
        return d


def pipeline(feature_set):
    added=FEATURES[feature_set];numeric=BASE+added
    preprocess=ColumnTransformer([
      ('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),numeric),
      ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=True),CATEGORIES)])
    return Pipeline([('risk_summary',TrainingRiskSummaries(include_dynamics=bool(set(added)&{'excursion','late_change'}),include_mean='mean_risk' in added)),
                     ('preprocess',preprocess),('model',Ridge(solver='lsqr'))])


def main(reuse_existing=False):
    for path in [OUT/'prediction',OUT/'tables',OUT/'figures',OUT/'reports']:path.mkdir(parents=True,exist_ok=True)
    p=pd.read_parquet(DATA_DIR / 'participants.parquet')
    w=pd.read_parquet(DATA_DIR / 'windows.parquet')
    assert p.participant_id.is_unique and w.window_row_id.is_unique
    assert w.loc[w.risk_raw.eq(0),'risk'].isna().all() and w.risk.dropna().between(1,10).all()
    rawprofile=w.groupby(['participant_id','scenario','window_index']).risk.mean().unstack(['scenario','window_index'])
    rawprofile.columns=[f'{b}_w{k}' for b,k in rawprofile.columns]
    rawprofile=rawprofile.reindex(columns=PROFILE)
    coverage=w.groupby('participant_id').risk.agg(['count','size'])
    rawprofile['positive_fraction']=coverage['count']/coverage['size']
    d=p.merge(rawprofile,on='participant_id',validate='one_to_one',how='left')
    assert len(d)==len(p)==2164 and d.positive_fraction.notna().all()
    for col in CATEGORIES:d[col]=d[col].fillna('Missing').astype(str)
    # X is explicitly limited to baseline covariates and within-task risk data.
    # No post-task trust or acceptance column can enter any estimator.
    x=d[BASE+CATEGORIES+PROFILE+['positive_fraction']].copy()
    assert not any('_post_' in c for c in x.columns)
    rng=np.random.default_rng(SEED)
    shuffled=rng.permutation(d.participant_id.to_numpy(copy=True));mapping={pid:i%5 for i,pid in enumerate(shuffled)}
    d['outer_person_fold']=d.participant_id.map(mapping)
    outputs=[];audit=[];peak_audit=[];existing=None
    if reuse_existing:
        prior=json.loads((OUT/'reports/a2_trust_prediction_metadata.json').read_text())
        for name in ['participants','windows']:assert digest(DATA_DIR/f'{name}.parquet')==prior['input_sha256'][name], 'Input changed; rerun without --reuse-existing'
        assert prior['seed']==SEED and prior['baseline_features']==BASE+CATEGORIES
        existing=pd.read_parquet(OUT/'prediction/a2_trust_oof.parquet')
        assert not existing.duplicated(['protocol','item','feature_set','participant_id']).any()
        for features in existing.feature_set.unique():assert FEATURES[features]==prior['feature_sets'][features]
        for (protocol,item,features),g in existing.groupby(['protocol','item','feature_set']):
            assert set(g.participant_id)==set(d.participant_id) and len(g)==len(d)
            assert np.array_equal(g.actual.to_numpy(),d.set_index('participant_id').loc[g.participant_id,f'trust_post_{item}'].to_numpy())
        outputs.append(existing)
    with threadpool_limits(2):
        for protocol in ['participant_5fold','leave_country_out']:
            labels=range(5) if protocol=='participant_5fold' else sorted(d.country.unique())
            for fold,label in enumerate(labels):
                test=(d.outer_person_fold==label).to_numpy() if protocol=='participant_5fold' else d.country.eq(label).to_numpy()
                train=~test;groups=d.loc[train,'participant_id'] if protocol=='participant_5fold' else d.loc[train,'country']
                train_ids=set(d.loc[train,'participant_id']);test_ids=set(d.loc[test,'participant_id'])
                assert not train_ids&test_ids
                if protocol=='leave_country_out':assert not set(d.loc[train,'country'])&set(d.loc[test,'country'])
                audit.append(dict(protocol=protocol,fold=fold,held_out=str(label),n_train=int(train.sum()),n_test=int(test.sum()),participant_overlap=0,
                    country_overlap=len(set(d.loc[train,'country'])&set(d.loc[test,'country'])),
                    train_id_sha256=hashlib.sha256(','.join(map(str,sorted(train_ids))).encode()).hexdigest(),
                    test_id_sha256=hashlib.sha256(','.join(map(str,sorted(test_ids))).encode()).hexdigest()))
                xt=x.loc[train];splits=list(GroupKFold(n_splits=3).split(xt,groups=groups))
                # Independent diagnostic of the actual training-only selectors.
                for stage,splitid,select in [('outer',-1,np.arange(len(xt)))]+[('inner',i,a) for i,(a,b) in enumerate(splits)]:
                    selector=TrainingRiskSummaries().fit(xt.iloc[select])
                    for block,peak in selector.peaks_.items():peak_audit.append(dict(protocol=protocol,outer_fold=fold,stage=stage,inner_fold=splitid,scenario=block,peak_position=peak,n_training=len(select),outer_test_used=False))
                for item in ['overall','acc','lks']:
                    y=d[f'trust_post_{item}']
                    assert y.notna().all() and y.between(0,10).all()
                    for feature_set in FEATURES:
                        if existing is not None:
                            reused=existing[(existing.protocol==protocol)&existing.item.eq(item)&existing.feature_set.eq(feature_set)&existing.fold.eq(fold)]
                            if len(reused):
                                assert set(reused.participant_id)==test_ids and len(reused)==len(test_ids), 'Cached predictions use different folds'
                                continue
                        fitted=GridSearchCV(pipeline(feature_set),{'model__alpha':[.1,1.,10.,100.]},cv=splits,scoring='neg_mean_squared_error',n_jobs=1)
                        fitted.fit(xt,y.loc[train]);pred=fitted.predict(x.loc[test])
                        outputs.append(pd.DataFrame(dict(participant_id=d.loc[test,'participant_id'],country=d.loc[test,'country'],actual=y.loc[test],pred=pred,
                            protocol=protocol,fold=fold,item=item,feature_set=feature_set,alpha=fitted.best_params_['model__alpha'])))
                print(f'finished {protocol} fold {fold+1}/{len(labels)}',flush=True)
    oof=pd.concat(outputs,ignore_index=True)
    assert not oof.duplicated(['protocol','item','feature_set','participant_id']).any()
    assert oof.groupby(['protocol','item','feature_set']).size().eq(len(d)).all()
    assert np.isfinite(oof.pred).all()
    oof.to_parquet(OUT/'prediction/a2_trust_oof.parquet',index=False)
    pd.DataFrame(audit).to_csv(OUT/'tables/a2_trust_cv_splits.csv',index=False)
    pd.DataFrame(peak_audit).to_csv(OUT/'tables/a2_trust_peak_selection.csv',index=False)
    # Same resampling weights for every model comparison; predictions stay fixed.
    weights=rng.multinomial(len(d),np.full(len(d),1/len(d)),size=2000)
    countries=sorted(d.country.unique());country_index=np.array([countries.index(c) for c in d.country]);country_draws=rng.multinomial(len(countries),np.full(len(countries),1/len(countries)),size=2000)
    clusterweights=country_draws[:,country_index]
    def delta_draws(y,error_difference,weight):
        mass=weight.sum(axis=1);sy=weight@y;sy2=weight@(y*y);denom=sy2-sy*sy/mass
        return np.divide(weight@error_difference,denom,out=np.full(len(weight),np.nan),where=denom>0)
    performance=[];records=[]
    for protocol in oof.protocol.unique():
        for item in ['overall','acc','lks']:
            paired=oof[(oof.protocol==protocol)&(oof.item==item)].pivot(index='participant_id',columns='feature_set',values='pred').reindex(d.participant_id)
            assert not paired.isna().any().any()
            y=d[f'trust_post_{item}'].to_numpy();sst=((y-y.mean())**2).sum();perf={}
            for features in FEATURES:
                prediction=paired[features].to_numpy();err=y-prediction
                row=dict(protocol=protocol,item=item,feature_set=features,n=len(d),r2=1-np.sum(err**2)/sst,rmse=np.sqrt(np.mean(err**2)),mae=np.mean(np.abs(err)))
                performance.append(row);perf[features]=row
            for reference,expanded in COMPARISONS:
                a=paired[reference].to_numpy();b=paired[expanded].to_numpy();ediff=(y-a)**2-(y-b)**2
                lo,hi=np.nanquantile(delta_draws(y,ediff,weights),[.025,.975]);clo=chi=np.nan
                if protocol=='leave_country_out':clo,chi=np.nanquantile(delta_draws(y,ediff,clusterweights),[.025,.975])
                records.append(dict(protocol=protocol,item=item,reference_feature_set=reference,feature_set=expanded,n=len(d),n_countries=len(countries),
                    r2_reference=perf[reference]['r2'],r2_expanded=perf[expanded]['r2'],delta_r2=perf[expanded]['r2']-perf[reference]['r2'],
                    delta_ci_low=lo,delta_ci_high=hi,delta_ci_low_country_cluster=clo,delta_ci_high_country_cluster=chi,
                    rmse_reference=perf[reference]['rmse'],rmse_expanded=perf[expanded]['rmse']))
    table=pd.DataFrame(records);performance=pd.DataFrame(performance)
    table.to_csv(OUT/'tables/a2_trust_cv_increment.csv',index=False);performance.to_csv(OUT/'tables/a2_trust_cv_performance.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    # Primary increment controls response coverage in the comparison model.
    fig,ax=plt.subplots(figsize=(6.4,3.8))
    for i,(protocol,label,color) in enumerate([('participant_5fold','New participants','#2563A6'),('leave_country_out','Held-out country','#D17A22')]):
        s=table[(table.protocol==protocol)&table.reference_feature_set.eq('pre_task_plus_response')&table.feature_set.eq('pre_task_plus_both')].set_index('item').reindex(['overall','acc','lks'])
        xx=np.arange(3)+(i-.5)*.15
        ax.errorbar(xx,s.delta_r2,yerr=np.vstack([s.delta_r2-s.delta_ci_low,s.delta_ci_high-s.delta_r2]),fmt='o',color=color,capsize=3,label=label)
    ax.axhline(0,color='#777777',lw=.8);ax.set(xticks=range(3),xticklabels=['Overall','Speed / distance','Lane centring'],ylabel='Added out-of-fold R²',title='Risk summaries beyond response coverage')
    ax.legend(frameon=False,fontsize=8);ax.text(.02,-.23,'Reference: all pre-task features + operated fraction\n95% intervals: paired participant bootstrap of fixed OOF predictions',transform=ax.transAxes,fontsize=8)
    fig.tight_layout()
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'figures/f5_added_trust_prediction.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    for protocol,label in [('participant_5fold','New participants'),('leave_country_out','Held-out country')]:
        fig,ax=plt.subplots(figsize=(6.3,4.0))
        for i,(feature,label2,color) in enumerate([('pre_task_plus_response','Response fraction','#9A718F'),('pre_task_plus_dynamics','Risk summaries','#16857D'),('pre_task_plus_both','Both','#2563A6')]):
            q=table[(table.protocol==protocol)&table.reference_feature_set.eq('pre_task')&table.feature_set.eq(feature)].set_index('item').reindex(['overall','acc','lks'])
            xx=np.arange(3)+(i-1)*.15;ax.errorbar(xx,q.delta_r2,yerr=[q.delta_r2-q.delta_ci_low,q.delta_ci_high-q.delta_r2],fmt='o',ms=4,capsize=2,color=color,label=label2)
        ax.axhline(0,color='#777777',lw=.8);ax.set(xticks=range(3),xticklabels=['Overall','Speed / distance','Lane centring'],ylabel='Added out-of-fold R² over pre-task model',title=f'{label}: feature comparison')
        ax.legend(frameon=False,fontsize=8);fig.tight_layout()
        for ext in ['pdf','svg','png']:fig.savefig(OUT/f'figures/f5_trust_feature_ablation_{protocol}.{ext}',dpi=200,bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.4,3.8))
    for i,(protocol,label,color) in enumerate([('participant_5fold','New participants','#2563A6'),('leave_country_out','Held-out country','#D17A22')]):
        q=table[(table.protocol==protocol)&table.reference_feature_set.eq('pre_task_plus_mean_risk_response')&table.feature_set.eq('pre_task_plus_both')].set_index('item').reindex(['overall','acc','lks'])
        xx=np.arange(3)+(i-.5)*.15;ax.errorbar(xx,q.delta_r2,yerr=[q.delta_r2-q.delta_ci_low,q.delta_ci_high-q.delta_r2],fmt='o',capsize=3,color=color,label=label)
    ax.axhline(0,color='#777777',lw=.8);ax.set(xticks=range(3),xticklabels=['Overall','Speed / distance','Lane centring'],ylabel='Added out-of-fold R²',title='Risk shape beyond mean risk and coverage')
    ax.legend(frameon=False,fontsize=8);ax.text(.02,-.23,'Added: excursion + late change, with training-only peak selection\n95% intervals: paired participant bootstrap of fixed OOF predictions',transform=ax.transAxes,fontsize=8);fig.tight_layout()
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'figures/f5_shape_above_mean.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    metadata={'status':'passed','n_participants':len(d),'n_countries':int(d.country.nunique()),'oof_rows':len(oof),
      'baseline_features':BASE+CATEGORIES,'feature_sets':FEATURES,'risk_features_computed_from':'participant-keyed 21-position profiles in current windows.parquet; no old feature CSV dependency',
      'peak_selection':'TrainingRiskSummaries.fit inside Pipeline in all inner and outer training fits; ties choose earliest questionnaire index',
      'outer_test_used_in_training':False,'post_task_fields_excluded_from_X':True,
      'interval_scope':'Fixed OOF paired participant bootstrap (2000 draws), no retraining or stimulus resampling; participant intervals condition on observed countries',
      'multiplicity':'Bootstrap intervals are pointwise exploratory intervals, not simultaneous intervals across outcomes, protocols or model comparisons; a difference in interval exclusion of zero is not a test of effect differences.',
      'metric_aggregation':'R2, RMSE and MAE pool all participant OOF predictions with equal participant weight; leave-country-out results are not equal-country-weighted averages of country-specific R2.',
      'country_cluster_interval_scope':'For leave_country_out only: resample 29 countries with replacement, retain all participant multiplicity within selected countries, use weighted SST; fixed OOF predictions and no model retraining',
      'interpretation':'Predictive increment, not a causal test; risk is measured during this same survey task. The baseline conditions on all three pre-trust and three pre-acceptance items; it differs from single-pre-item OLS association models.',
      'input_sha256':{name:digest(DATA_DIR/f'{name}.parquet') for name in ['participants','windows']},'script_sha256':digest(Path(__file__)),'seed':SEED,
      'feature_missingness_before_training_imputation':TrainingRiskSummaries().fit(x).transform(x)[DYNAMICS+['positive_fraction']].isna().sum().to_dict(),
      'primary_comparison':'pre_task_plus_both minus pre_task_plus_response separates risk summary increment from response coverage',
      'shape_comparison':'pre_task_plus_both minus pre_task_plus_mean_risk_response adds excursion and late change beyond mean risk and response fraction',
      'reused_existing_oof_rows':0 if existing is None else len(existing),
      'clipping':'No post hoc clipping of predictions'}
    (OUT/'reports/a2_trust_prediction_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(table.to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reuse-existing',action='store_true',help='Reuse complete, source- and fold-verified OOF groups and fit newly added groups only.')
    main(parser.parse_args().reuse_existing)
