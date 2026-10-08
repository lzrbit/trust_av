"""Exploratory positive-response dynamics, all-country comparisons and trust links.

Risk zero is non-operation. Trust/acceptance zero remains valid. All intervals
condition on the observed stimulus library; no physical timing is inferred.
"""
from settings import WORKSPACE as ROOT, DATA_DIR, QUESTIONNAIRE, REPO, relative_label
from pathlib import Path
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
from scipy import stats, optimize
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


OUT=ROOT/'outputs/core'
TABLES=OUT/'tables'; FIGS=OUT/'figures'; REPORTS=OUT/'reports'
SEED=20261012; B=2000
BLOCKS=['B1','B2','B3','B4']; NPOS={'B1':6,'B2':5,'B3':5,'B4':5}
COLORS=['#2563A6','#D17A22','#16857D','#8B4B9E']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})

def short_country(country):
    return {'United Kingdom of Great Britain and Northern Ireland':'United Kingdom','United States of America':'United States'}.get(country,country)

def savefig(fig,name):
    fig.tight_layout()
    for ext in ['pdf','svg','png']:
        fig.savefig(FIGS/f'{name}.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)

def save(frame,name):
    frame.to_csv(TABLES/f'a2_{name}.csv',index=False)

def bootstrap_means(values,seed=SEED,b=B):
    a=np.asarray(values,dtype=float)
    if a.ndim==1:a=a[:,None]
    rng=np.random.default_rng(seed)
    weight=rng.multinomial(len(a),np.full(len(a),1/len(a)),size=b)
    numer=weight@np.nan_to_num(a,nan=0.)
    denom=weight@np.isfinite(a).astype(float)
    with np.errstate(invalid='ignore',divide='ignore'):
        draws=numer/denom
    return np.nanquantile(draws,[.025,.975],axis=0),draws

def bh(frame,pcol='p',qcol='q',family='family'):
    frame[qcol]=np.nan
    for _,idx in frame.groupby(family).groups.items():
        valid=frame.loc[idx,pcol].notna()
        ids=frame.loc[idx].index[valid]
        if len(ids):frame.loc[ids,qcol]=multipletests(frame.loc[ids,pcol],method='fdr_bh')[1]
    return frame

def profile_matrix(w,ids,value='risk'):
    ans=w.groupby(['participant_id','scenario','window_index'])[value].mean().unstack(['scenario','window_index'])
    columns=pd.MultiIndex.from_tuples([(block,k) for block in BLOCKS for k in range(1,NPOS[block]+1)])
    return ans.reindex(index=ids,columns=columns)

def paired_record(pre,post):
    valid=np.isfinite(pre)&np.isfinite(post)
    a=np.asarray(post)[valid]-np.asarray(pre)[valid]
    if len(a)<2:return dict(n=len(a),delta=np.nan,ci_low=np.nan,ci_high=np.nan,p=np.nan,dz=np.nan)
    ci,_=bootstrap_means(a)
    sd=a.std(ddof=1)
    pv=stats.ttest_1samp(a,0).pvalue if sd>0 else (1. if np.all(a==0) else np.nan)
    return dict(n=len(a),delta=a.mean(),ci_low=ci[0,0],ci_high=ci[1,0],p=max(pv,np.finfo(float).tiny) if np.isfinite(pv) else pv,dz=a.mean()/sd if sd>0 else np.nan)

def normal_meta(raw):
    """Normal-normal exploratory shrinkage, with explicitly regularized variances.

    Country sampling variances borrow 8 degrees of freedom from the pooled
    within-country variance. Conditional intervals omit tau estimation error.
    """
    r=raw.copy(); n=r.n.to_numpy(float); y=r.delta.to_numpy(float)
    pool=np.nansum((n-1)*r.sd.to_numpy(float)**2)/max(np.sum(n-1),1)
    if not np.isfinite(pool) or pool<=0:pool=1.
    within=((n-1)*np.nan_to_num(r.sd.to_numpy(float),nan=0.)**2+8*pool)/(n-1+8)
    v=within/n
    def objective(tau):
        total=v+tau; weight=1/total; mu=np.sum(weight*y)/sum(weight)
        return .5*(np.log(total).sum()+np.log(weight.sum())+(weight*(y-mu)**2).sum())
    opt=optimize.minimize_scalar(objective,bounds=(0,max(4.,np.var(y)*4)),method='bounded')
    tau=0. if objective(0)<=opt.fun else float(opt.x)
    ww=1/(v+tau); mu=float(np.sum(ww*y)/sum(ww)); varmu=1/sum(ww)
    shrink=tau/(v+tau)
    posterior=shrink*y+(1-shrink)*mu
    vp=tau*v/(tau+v)+(1-shrink)**2*varmu
    r['sampling_variance_regularized']=v; r['pooled_within_variance']=pool
    r['tau2_reml']=tau; r['grand_mean']=mu; r['shrink_weight']=shrink
    r['estimate_eb']=posterior; r['ci_low_eb']=posterior-1.96*np.sqrt(vp);r['ci_high_eb']=posterior+1.96*np.sqrt(vp)
    r['raw_ci_low_normal']=y-1.96*np.sqrt(v);r['raw_ci_high_normal']=y+1.96*np.sqrt(v)
    r['interval_scope']='conditional normal-normal approximation; 8-df pooled variance; tau uncertainty omitted'
    return r

def main():
    for d in [TABLES,FIGS,REPORTS]:d.mkdir(parents=True,exist_ok=True)
    p=pd.read_parquet(DATA_DIR / 'participants.parquet')
    e=pd.read_parquet(DATA_DIR / 'events.parquet')
    w=pd.read_parquet(DATA_DIR / 'windows.parquet')
    p=p.set_index('participant_id',drop=False)
    assert p.index.is_unique and (w.risk.dropna()>0).all()
    w['risk']=w.risk.astype(float)
    mat=profile_matrix(w,p.index)
    p=p.loc[mat.index]
    ci,_=bootstrap_means(mat)
    means=np.nanmean(mat,axis=0)
    rows=[]
    for j,(block,position) in enumerate(mat.columns):
        rows.append(dict(scenario=block,position=position,n_participants=int(mat.iloc[:,j].notna().sum()),mean=means[j],ci_low=ci[0,j],ci_high=ci[1,j]))
    summary=pd.DataFrame(rows);save(summary,'position_means')
    contrasts=[]
    for block in BLOCKS:
        for first,last in [(k,k+1) for k in range(1,NPOS[block])]+[(1,NPOS[block])]:
            contrasts.append(dict(scenario=block,first=first,last=last,family='21_position_contrasts',**paired_record(mat[(block,first)],mat[(block,last)])))
    contrasts=bh(pd.DataFrame(contrasts));save(contrasts,'position_contrasts')
    for block,color in zip(BLOCKS,COLORS):
        s=summary[summary.scenario==block]
        fig,ax=plt.subplots(figsize=(4.5,3.5))
        ax.errorbar(s.position,s['mean'],yerr=np.vstack([s['mean']-s.ci_low,s.ci_high-s['mean']]),color=color,marker='o',capsize=3)
        ax.set(xlabel='Questionnaire rating position',ylabel='Mean operated risk rating',title=block,ylim=(1,10),xticks=s.position)
        ax.text(.02,.97,'0 = non-operation; observed scores 1-10',transform=ax.transAxes,va='top',fontsize=8,color='#555555')
        savefig(fig,f'f2_risk_trajectory_{block}')

    # Cross-fit the descriptive peak location, rather than treating a selected
    # pooled maximum as an a priori test. Folds group participants throughout.
    rng=np.random.default_rng(SEED); ids=p.index.to_numpy(copy=True); rng.shuffle(ids)
    foldmap={pid:k for k,part in enumerate(np.array_split(ids,5)) for pid in part}
    folds=np.array([foldmap[i] for i in mat.index]); peakrows=[]
    excursions=pd.DataFrame(index=mat.index)
    latechanges=pd.DataFrame(index=mat.index)
    for block in BLOCKS:
        for k in range(5):
            train=mat.loc[folds!=k,block]; test=mat.loc[folds==k,block]
            peak=int(train.mean().idxmax())
            excursions.loc[test.index,block]=test[peak]-test[1]
            latechanges.loc[test.index,block]=test[NPOS[block]]-test[peak]
            peakrows.append(dict(scenario=block,fold=k,peak_position_from_training=peak,n_test=len(test)))
    save(pd.DataFrame(peakrows),'crossfit_peak_locations')
    curvefeatures=pd.DataFrame({'participant_id':mat.index,
       'mean_risk':mat.mean(axis=1),'excursion':excursions.mean(axis=1),
       'late_change':latechanges.mean(axis=1),
       'positive_fraction':w.groupby('participant_id').risk.count().reindex(mat.index)/w.groupby('participant_id').size().reindex(mat.index)})
    curvefeatures=curvefeatures.set_index('participant_id',drop=False)
    assert np.array_equal(p.index.to_numpy(),p.participant_id.to_numpy()), 'Participant index changed'
    source_profile=profile_matrix(w,p.index)
    assert np.allclose(curvefeatures.mean_risk.to_numpy(),source_profile.mean(axis=1).to_numpy(),equal_nan=True)
    save(curvefeatures.reset_index(drop=True),'participant_risk_features')

    # Every country remains visible. Similarity is descriptive, conditional on
    # available positive responses, and cannot identify cultural mechanisms.
    country_order=p.country.value_counts().index.tolist()
    bycountry=mat.groupby(p.country).mean().reindex(country_order)
    ncountry=p.country.value_counts().reindex(country_order)
    centered=bycountry.copy()
    for block in BLOCKS:centered.loc[:,block]=bycountry[block].sub(bycountry[block].mean(axis=1),axis=0).values
    country_profiles=bycountry.copy();country_profiles.columns=[f'{a}_w{b}' for a,b in country_profiles.columns]
    country_profiles.insert(0,'n_participants',ncountry);country_profiles.insert(0,'country',country_profiles.index)
    save(country_profiles.reset_index(drop=True),'country_profiles')
    fig,ax=plt.subplots(figsize=(10, max(5,len(country_order)*.24)))
    vmax=max(1,float(np.nanmax(np.abs(centered.to_numpy()))))
    im=ax.imshow(centered.to_numpy(),aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
    ax.set_yticks(range(len(country_order)),[f'{short_country(c)} (n={ncountry[c]})' for c in country_order],fontsize=8)
    ax.set_xticks(range(21),[f'{b}:{k}' for b,k in centered.columns],rotation=90,fontsize=8)
    ax.set(xlabel='Block : rating position',title='Country profiles after removing each block mean')
    for cut in [5.5,10.5,15.5]:ax.axvline(cut,color='white',lw=1.5)
    fig.colorbar(im,ax=ax,label='Deviation from country-block mean')
    savefig(fig,'f2_country_shape_heatmap')

    similarity=[]; countrycontrasts=[]
    for ci_country,country in enumerate(country_order):
        mask=p.country.eq(country).to_numpy(); local=mat.loc[mask]
        external=mat.loc[~mask].mean()
        reference=external.copy(); prof=local.mean()
        for block in BLOCKS:
            reference.loc[block]=(external[block]-external[block].mean()).to_numpy()
            prof.loc[block]=(prof[block]-prof[block].mean()).to_numpy()
            peak=int(external[block].idxmax())
            diff=(local[(block,peak)]-local[(block,1)]).dropna()
            if len(diff):countrycontrasts.append(dict(country=country,scenario=block,peak_position_outside_country=peak,n=len(diff),delta=diff.mean(),sd=diff.std(ddof=1)))
        good=np.isfinite(prof)&np.isfinite(reference)
        rval=stats.pearsonr(prof[good],reference[good]).statistic if good.sum()>3 and np.std(prof[good])>0 else np.nan
        lo=hi=np.nan; valid_draws=0
        if len(local)>=2:
            _,draws=bootstrap_means(local,seed=SEED+ci_country,b=1000)
            offset=0
            for block in BLOCKS:
                q=NPOS[block];draws[:,offset:offset+q]-=np.nanmean(draws[:,offset:offset+q],axis=1,keepdims=True);offset+=q
            ref=reference.to_numpy();d=draws[:,good];r=ref[good]
            d-=np.mean(d,axis=1,keepdims=True);r=r-r.mean()
            denom=np.linalg.norm(d,axis=1)*np.linalg.norm(r)
            rd=np.divide(d@r,denom,out=np.full(len(d),np.nan),where=denom>0)
            valid_draws=int(np.isfinite(rd).sum())
            lo,hi=np.nanquantile(rd,[.025,.975])
        similarity.append(dict(country=country,n_participants=len(local),shape_r=rval,ci_low=lo,ci_high=hi,bootstrap_attempted=1000 if len(local)>=2 else 0,bootstrap_valid=valid_draws,interval='within-country participant bootstrap; external template fixed' if len(local)>=2 else 'not estimable from one participant'))
    similarity=pd.DataFrame(similarity);save(similarity,'country_shape_similarity')
    fig,ax=plt.subplots(figsize=(5.8,max(5,len(similarity)*.24)))
    y=np.arange(len(similarity))
    for i,r in similarity.iterrows():
        color='#2563A6' if r.n_participants>=30 else '#777777'
        if np.isfinite(r.ci_low):ax.plot([r.ci_low,r.ci_high],[i,i],color=color,lw=1)
        ax.scatter(r.shape_r,i,s=18,color=color)
    ax.axvline(0,color='#aaaaaa',lw=.8);ax.set_yticks(y,[f'{short_country(r.country)} (n={r.n_participants})' for _,r in similarity.iterrows()],fontsize=8)
    ax.invert_yaxis();ax.set(xlim=(-1.05,1.05),xlabel='Within-block-centred profile correlation',title='Similarity to the other-country profile')
    savefig(fig,'f2_country_shape_similarity')
    eb=[]
    for block in BLOCKS:
        raw=pd.DataFrame(countrycontrasts); fit=normal_meta(raw[raw.scenario==block])
        eb.append(fit)
        s=fit.set_index('country').reindex(country_order).reset_index()
        fig,ax=plt.subplots(figsize=(6.2,max(5,len(s)*.24)))
        for i,r in s.iterrows():
            if not np.isfinite(r.delta):continue
            ax.plot([r.raw_ci_low_normal,r.raw_ci_high_normal],[i,i],color='#c5c5c5',lw=.9)
            ax.scatter(r.delta,i,color='#aaaaaa',marker='x',s=18)
            ax.plot([r.ci_low_eb,r.ci_high_eb],[i+.12,i+.12],color='#2563A6',lw=1)
            ax.scatter(r.estimate_eb,i+.12,color='#2563A6',s=15)
        ax.set_yticks(range(len(s)),[f'{short_country(r.country)} (paired n={int(r.n)})' if np.isfinite(r.n) else r.country for _,r in s.iterrows()],fontsize=8)
        ax.invert_yaxis();ax.axvline(0,color='black',lw=.6)
        ax.set(xlabel='Other-country-selected high point minus first rating',title=f'{block}: all-country estimates with partial pooling')
        ax.text(.02,-.095,'Grey: unpooled points + regularized intervals; blue: partial pooling.',transform=ax.transAxes,fontsize=7)
        savefig(fig,f'f2_country_excursion_{block}')
    save(pd.concat(eb,ignore_index=True),'country_excursion_partial_pooling')

    # Sensitivities are explicit alternative estimands/cohorts, not substitute
    # primary analyses: treating zeros as scores is included only as diagnostic.
    variants={'positive_primary':mat,'zeros_as_scores_diagnostic':profile_matrix(w,p.index,'risk_raw')}
    if 'qc_duration' in p:variants['positive_duration_qc']=mat.loc[p.qc_duration.fillna(False)]
    complete_ids=w.groupby('event_row_id').risk.apply(lambda a:a.notna().all())
    wc=w[w.event_row_id.isin(complete_ids[complete_ids].index)]
    variants['positive_complete_events']=profile_matrix(wc,p.index)
    sens=[]
    for name,a in variants.items():
        for (block,pos),v in a.mean().items():sens.append(dict(variant=name,scenario=block,position=pos,mean=v,n_participants=int(a[(block,pos)].notna().sum())))
    for (block,pos),v in bycountry.mean(axis=0).items():sens.append(dict(variant='positive_equal_country_weight',scenario=block,position=pos,mean=v,n_participants=len(p)))
    sens=pd.DataFrame(sens);save(sens,'sensitivity_profiles')
    for block in BLOCKS:
        fig,ax=plt.subplots(figsize=(5.6,3.7))
        for name,g in sens[sens.scenario==block].groupby('variant'):
            ax.plot(g.position,g['mean'],marker='o',ms=3,label=name.replace('positive_','').replace('_',' '),ls='--' if name.startswith('zeros') else '-')
        ax.set(xlabel='Questionnaire rating position',ylabel='Mean rating',title=f'{block}: specified sensitivities',ylim=(0,10))
        ax.legend(fontsize=7,frameon=False,loc='upper left')
        savefig(fig,f'f6_risk_sensitivity_{block}')

    # Separate trust questions and their within-person differential update.
    trust=[]
    for item in ['overall','acc','lks']:
        trust.append(dict(item=item,family='trust_changes',**paired_record(p[f'trust_pre_{item}'],p[f'trust_post_{item}'])))
    tr=bh(pd.DataFrame(trust));save(tr,'trust_changes')
    double=(p.trust_post_lks-p.trust_pre_lks)-(p.trust_post_acc-p.trust_pre_acc)
    diff=paired_record(np.zeros(len(double)),double)
    country_trust=[]
    for country in country_order:
        vals=double.loc[p.country==country].dropna()
        country_trust.append(dict(country=country,n=len(vals),delta=vals.mean(),sd=vals.std(ddof=1)))
    country_trust=normal_meta(pd.DataFrame(country_trust));save(country_trust,'trust_differential_country')
    df=p.copy()
    for col in ['mean_risk','excursion','late_change','positive_fraction']:df[col]=curvefeatures[col]
    predictors=['mean_risk','excursion','late_change','positive_fraction']
    for col in predictors:df[col+'_z']=(df[col]-df[col].mean())/df[col].std(ddof=1)
    df['age_10']=(df.age-df.age.mean())/10;df['experience_10']=(df.driving_exp_years-df.driving_exp_years.mean())/10
    df['gender']=df.gender.fillna('Missing').astype(str);df['country']=df.country.fillna('Unknown').astype(str)
    regrows=[]; modelmeta=[]
    for item in ['overall','acc','lks']:
        formula=f'trust_post_{item} ~ trust_pre_{item} + mean_risk_z + excursion_z + late_change_z + positive_fraction_z + age_10 + experience_10 + C(gender) + C(country)'
        model=smf.ols(formula,data=df).fit(cov_type='HC1')
        modelmeta.append(dict(outcome=item,formula=formula,n=int(model.nobs),r_squared_in_sample=model.rsquared,covariance='HC1; one row per participant; countries fixed; associations only'))
        ci=model.conf_int()
        for col in predictors:
            term=col+'_z'
            regrows.append(dict(item=item,feature=col,family='12_risk_trust_associations',estimate=model.params[term],ci_low=ci.loc[term,0],ci_high=ci.loc[term,1],p=model.pvalues[term],n=int(model.nobs)))
    regression=bh(pd.DataFrame(regrows));save(regression,'trust_risk_associations')
    fig,ax=plt.subplots(figsize=(6.6,4.4))
    for k,item in enumerate(['overall','acc','lks']):
        s=regression[regression.item==item].set_index('feature').reindex(predictors)
        y=np.arange(len(predictors))+(k-1)*.20
        ax.errorbar(s.estimate,y,xerr=np.vstack([s.estimate-s.ci_low,s.ci_high-s.estimate]),fmt='o',ms=4,capsize=2,label=item.upper(),color=COLORS[k])
    ax.axvline(0,color='black',lw=.7);ax.set_yticks(range(4),['Mean risk','Cross-fit excursion','Cross-fit late change','Operated fraction'])
    ax.invert_yaxis();ax.set(xlabel='Adjusted post-task trust association per predictor SD',title='Risk dynamics and function-specific trust')
    ax.legend(frameon=False,fontsize=8);savefig(fig,'f5_risk_trust_associations')
    fig,ax=plt.subplots(figsize=(5.2,3.5))
    x=np.arange(3);ax.errorbar(x,tr.delta,yerr=np.vstack([tr.delta-tr.ci_low,tr.ci_high-tr.delta]),fmt='o',capsize=4,color='#2563A6')
    ax.axhline(0,color='#888888',lw=.7);ax.set(xticks=x,xticklabels=['Overall','Speed / distance','Lane centring'],ylabel='Post minus pre trust (0-10 scale)',title='Function-specific trust updates')
    savefig(fig,'f5_trust_updates')
    fig,ax=plt.subplots(figsize=(6.2,max(5,len(country_trust)*.24)))
    s=country_trust.set_index('country').reindex(country_order).reset_index()
    for i,r in s.iterrows():
        ax.plot([r.ci_low_eb,r.ci_high_eb],[i,i],color='#2563A6',lw=1);ax.scatter(r.estimate_eb,i,color='#2563A6',s=16);ax.scatter(r.delta,i,color='#999999',marker='x',s=18)
    ax.axvline(0,color='black',lw=.7);ax.invert_yaxis();ax.set_yticks(range(len(s)),[f'{short_country(r.country)} (n={r.n})' for _,r in s.iterrows()],fontsize=8)
    ax.set(xlabel='LKS trust change minus ACC trust change',title='Differential updating across all countries')
    savefig(fig,'f5_country_trust_differential')
    metadata={'n_participants':len(p),'n_events':len(e),'n_countries':int(p.country.nunique()),'n_positive_ratings':int(w.risk.notna().sum()),
       'bootstrap_draws':B,'seed':SEED,'risk_zero':'non-operation; missing','prepost_zero':'valid',
       'trust_differential':diff,'trust_models':modelmeta,
       'country_pooling':'Exploratory normal-normal REML, within-country variances regularized using 8 df pooled variance; conditional intervals omit tau uncertainty.',
       'limits':['Observed-stimulus conditional inference','Operated-response conditional distribution; no claim missing at random','Position indices, not verified physical time','Country differences not cultural causes','No causal inference from trust associations'],
       'input_sha256':{name:hashlib.sha256((DATA_DIR/f'{name}.parquet').read_bytes()).hexdigest() for name in ['participants','events','windows']}}
    (REPORTS/'a2_dynamics_metadata.json').write_text(json.dumps(metadata,indent=2,default=lambda x:float(x)))
    assert len(summary)==21 and len(contrasts)==21 and len(similarity)==p.country.nunique()
    assert similarity.shape_r.notna().all(), 'Country profile similarity unexpectedly missing'
    assert np.isfinite(centered.to_numpy()).all(), 'Country heatmap unexpectedly missing'
    assert all(len(x)==p.country.nunique() for x in eb)
    (REPORTS/'a2_validation.json').write_text(json.dumps({'status':'passed','all_countries_retained':True,'positive_risk_only':True,'position_cells':21,'paired_contrasts':21,'all_trust_zeros_retained':True},indent=2))
    print(json.dumps({'counts':metadata['n_positive_ratings'],'countries':metadata['n_countries'],'trust_differential':diff},default=lambda x:float(x)),flush=True)

if __name__=='__main__':
    main()
