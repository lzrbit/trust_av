"""Independent descriptive panels for NHB round 2; all countries, risk > 0.

Confidence intervals resample participants with replacement, preserving duplicate
draws. No item-transition bins are fitted or interpreted as latent categories.
"""
from __future__ import annotations
from settings import WORKSPACE as ROOT, DATA_DIR, QUESTIONNAIRE, REPO, relative_label
import argparse
import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MPath
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, to_tree, leaves_list


NAVY='#243D53'; TEAL='#268E8D'; GOLD='#DCAB51'; RUST='#BD6454'; GREY='#A4ACB3'
ITEMS=[('trust','overall','Overall trust'),('trust','acc','ACC trust'),('trust','lks','LKS trust'),
       ('accept','delegate','Delegate driving'),('accept','monitor_raw','Need for supervision'),('accept','distract','Non-driving activities')]
SHORT={'United Kingdom of Great Britain and Northern Ireland':'United Kingdom','United States of America':'United States','Czech Republic':'Czech Republic'}

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def boot_mean(a,rng,b=1000):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    if len(a)==0:return np.nan,np.nan,np.nan,0
    if len(a)==1:return float(a[0]),np.nan,np.nan,1
    draws=np.mean(a[rng.integers(0,len(a),size=(b,len(a)))],axis=1)
    lo,hi=np.quantile(draws,[.025,.975]);return float(a.mean()),float(lo),float(hi),len(a)

def bh(p):
    p=np.asarray(p,float);order=np.argsort(p);ranked=p[order]
    q=np.minimum.accumulate((ranked*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    out=np.empty(len(p));out[order]=np.minimum(q,1);return out

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,
        'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#697580','axes.labelcolor':NAVY,'text.color':NAVY,
        'xtick.color':'#51616F','ytick.color':'#51616F','svg.fonttype':'none',
        'pdf.fonttype':42,'savefig.facecolor':'white','figure.facecolor':'white'})

def main(root,bootstrap,tree_bootstrap,seed):
    out=root/'outputs/core';data=DATA_DIR;tab=out/'tables';figdir=out/'figures';rep=out/'reports'
    for x in [tab,figdir,rep]:x.mkdir(parents=True,exist_ok=True)
    inputs={name:digest(data/f'{name}.parquet') for name in ['participants','events','windows','source_participants']}
    p=pd.read_parquet(data/'participants.parquet');e=pd.read_parquet(data/'events.parquet');w=pd.read_parquet(data/'windows.parquet');s=pd.read_parquet(data/'source_participants.parquet')
    rng=np.random.default_rng(seed);style();captions=[];files=[]
    def save(fig,name,caption):
        for suffix in ['pdf','svg','png']:
            path=figdir/f'{name}.{suffix}';fig.savefig(path,bbox_inches='tight',dpi=220);files.append(path.name)
        plt.close(fig);captions.append({'figure':name,'caption':caption})

    # Separate denominators keep zero-coded ratings out of participant exclusions.
    fig,ax=plt.subplots(figsize=(6.4,4.0));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    flow=[(.03,.70,.43,.22,f'{len(s):,} respondents\nRaw questionnaire export',NAVY),
          (.03,.36,.43,.22,f'{len(p):,} participants\nCompleted + licensed\n{p.country.nunique()} countries retained',TEAL),
          (.54,.70,.43,.22,f'{len(w):,} window slots\n{len(e):,} assigned events',NAVY),
          (.54,.36,.43,.22,f'{w.risk.notna().sum():,} valid ratings\nAcross {e.risk_event_mean.notna().sum():,} events\nRisk score > 0',TEAL)]
    for x,y,ww,hh,txt,c in flow:
        ax.add_patch(FancyBboxPatch((x,y),ww,hh,boxstyle='round,pad=.008,rounding_size=.015',facecolor=c,edgecolor='none'))
        ax.text(x+ww/2,y+hh/2,txt,ha='center',va='center',color='white',fontsize=9)
    for x in [.05,.56]:ax.annotate('',xy=(x,.60),xytext=(x,.69),arrowprops={'arrowstyle':'->','color':NAVY,'lw':1.5})
    ax.text(.27,.635,f'{len(s)-len(p):,} incomplete / unlicensed',ha='center',fontsize=8,color=RUST)
    ax.text(.78,.635,f'{w.risk_raw.eq(0).sum():,} zeros = no operation',ha='center',fontsize=8,color=RUST)
    ax.text(.5,.19,'All 2,164 retained participants have valid risk data.\nDuration flags are reserved for sensitivity analysis.',ha='center',va='center',fontsize=9)
    ax.set_title('Participants and observed ratings',loc='left',pad=10)
    save(fig,'f1_a_sample_flow','All completed/licensed participants are retained. Risk 0 is no operation and excluded only from the affected risk observation. Pre/post trust and acceptance zeros remain valid. Participant and rating flows have different denominators.')

    co=p.groupby('country',dropna=False).size().rename('n_completed_licensed').reset_index().sort_values('n_completed_licensed',ascending=True)
    co['small_n_flag']=co.n_completed_licensed.lt(30)
    fig,ax=plt.subplots(figsize=(6.4,7.4));y=np.arange(len(co))
    ax.barh(y,co.n_completed_licensed,color=np.where(co.small_n_flag,GREY,TEAL),height=.73)
    ax.set_yticks(y,[(SHORT.get(c,c)+' †' if n<30 else SHORT.get(c,c)) for c,n in zip(co.country,co.n_completed_licensed)],fontsize=8)
    for yi,n in zip(y,co.n_completed_licensed):ax.text(n+7,yi,str(n),va='center',fontsize=8)
    ax.set_xlim(0,co.n_completed_licensed.max()*1.13);ax.set_xlabel('Participants');ax.set_title('All reported countries retained',loc='left')
    fig.text(.125,.018,'† n < 30: descriptive flag; no exclusion',fontsize=8,color='#667580')
    save(fig,'f1_b_country_counts','All 29 countries are retained. Grey bars and dagger symbols flag n<30 for interpretation; this is not an eligibility threshold. Country-specific valid and no-operation rating counts are in d2_country_coverage.csv.')

    dem=[]
    for field,label,bins,name in [('age','Age (years)',np.arange(17.5,75,3),'f1_c_age_distribution'),('driving_exp_years','Driving experience (years)',np.arange(-.5,58,3),'f1_d_experience_distribution')]:
        a=p[field].dropna();fig,ax=plt.subplots(figsize=(4.8,3.6));ax.hist(a,bins=bins,color=TEAL,edgecolor='white',linewidth=.5)
        ax.axvline(a.median(),color=NAVY,lw=1.2,ls='--');ax.set(xlabel=label,ylabel='Participants',title=label.replace(' (years)',''))
        ax.text(.97,.96,f'n = {len(a):,}\nMedian = {a.median():.0f}\nIQR = {a.quantile(.25):.0f}–{a.quantile(.75):.0f}',transform=ax.transAxes,ha='right',va='top',fontsize=9)
        dem.append({'variable':field,'n':len(a),'mean':a.mean(),'sd':a.std(),'median':a.median(),'q25':a.quantile(.25),'q75':a.quantile(.75),'min':a.min(),'max':a.max()})
        save(fig,name,'Original questionnaire age/experience at the 2023 survey is retained. Dashed line is the median; no age-based participant exclusions are applied.')
    pd.DataFrame(dem).to_csv(tab/'d2_demographics.csv',index=False)
    license_age=p.age-p.driving_exp_years
    pd.DataFrame([
        {'check':'estimated_age_at_licence_below_16','n_flagged':int(license_age.lt(16).sum()),'rule':'age minus driving_exp_years < 16','role':'QC only; jurisdiction-specific legality not assessed; no primary exclusion'},
        {'check':'estimated_age_at_licence_nonpositive','n_flagged':int(license_age.le(0).sum()),'rule':'age minus driving_exp_years <= 0','role':'Internally implausible source response; no silent recoding or primary exclusion'},
        {'check':'missing_gender','n_flagged':int(p.gender.isna().sum()),'rule':'source gender absent','role':'Retained as Not reported for descriptive plots'}
    ]).to_csv(tab/'d2_demographic_quality_flags.csv',index=False)
    fig,ax=plt.subplots(figsize=(4.7,3.9));h=ax.hexbin(p.age,p.driving_exp_years,gridsize=24,mincnt=1,cmap='viridis',linewidths=.1)
    fig.colorbar(h,ax=ax,label='Participants');ax.set(xlabel='Age (years)',ylabel='Driving experience (years)',title='Age and driving experience')
    save(fig,'f1_e_age_experience','Each participant contributes once. Hexagon colour is the number of respondents. Both age and driving experience use the original survey values.')

    gres=[]
    gp=p.assign(gender_display=p.gender.fillna('Not reported'))
    for label,g in gp.groupby('gender_display'):
        m,lo,hi,n=boot_mean(g.risk_person_block_balanced_mean,rng,bootstrap);gres.append({'gender':label,'n':n,'risk_mean':m,'ci_low':lo,'ci_high':hi})
    gdf=pd.DataFrame(gres);gdf.to_csv(tab/'d2_gender_risk_descriptive.csv',index=False)
    fig,ax=plt.subplots(figsize=(4.8,3.6));y=np.arange(len(gdf))
    ax.errorbar(gdf.risk_mean,y,xerr=[gdf.risk_mean-gdf.ci_low,gdf.ci_high-gdf.risk_mean],fmt='o',color=TEAL,ecolor=NAVY,capsize=3)
    ax.set_yticks(y,[f'{z.gender} (n={z.n:,})' for z in gdf.itertuples()]);ax.set(xlim=(1,10),xlabel='Participant mean risk (1–10)',title='Descriptive risk by reported gender');ax.invert_yaxis()
    save(fig,'f1_f_gender_risk','Participant block-balanced risk is the mean of four within-block means of valid event means. Points are descriptive group means and bars are participant-bootstrap 95% intervals. No covariate adjustment or causal gender interpretation is applied. Not-reported gender remains a separate descriptive group.')

    itemrows=[]
    for fam,item,label in ITEMS:
        pre=p[f'{fam}_pre_{item}'].to_numpy(float);post=p[f'{fam}_post_{item}'].to_numpy(float)
        mask=np.isfinite(pre)&np.isfinite(post);pre=pre[mask];post=post[mask];ind=rng.integers(0,len(pre),size=(bootstrap,len(pre)))
        for metric,a in [('pre',pre),('post',post),('delta',post-pre)]:
            draws=a[ind].mean(axis=1);lo,hi=np.quantile(draws,[.025,.975]);itemrows.append({'family':fam,'item':item,'label':label,'metric':metric,'n':len(a),'mean':a.mean(),'ci_low':lo,'ci_high':hi,'zero_is_valid':True})
    idf=pd.DataFrame(itemrows);idf.to_csv(tab/'d2_item_prepost.csv',index=False)
    fig,ax=plt.subplots(figsize=(6.3,4.1))
    for i,(_,_,label) in enumerate(ITEMS):
        q=idf[(idf.label==label)&idf.metric.isin(['pre','post'])].set_index('metric');ax.plot(q.loc[['pre','post'],'mean'],[i,i],color=GREY,lw=2)
        for metric,c,offset in [('pre',NAVY,-.065),('post',TEAL,.065)]:
            row=q.loc[metric];ax.errorbar(row['mean'],i+offset,xerr=[[row['mean']-row.ci_low],[row.ci_high-row['mean']]],fmt='o',color=c,capsize=2,label=metric.capitalize() if i==0 else None)
    ax.set_yticks(range(6),[x[2] for x in ITEMS]);ax.set(xlim=(0,10),xlabel='Original item score (0–10)',title='Before and after video evaluation');ax.invert_yaxis();ax.legend(frameon=False,loc='lower right',ncol=2)
    save(fig,'f1_g_item_prepost','Six original items, each on 0–10 with zero valid. Points are pre/post means, with participant-bootstrap 95% intervals. Supervision remains in its original direction: higher values indicate greater need to monitor. Lines connect sample means, not latent constructs or individual trajectories.')
    d=idf[idf.metric.eq('delta')].copy();fig,ax=plt.subplots(figsize=(6.2,4.1));y=np.arange(6)
    ax.axvline(0,color=GREY,lw=1);ax.errorbar(d['mean'],y,xerr=[d['mean']-d.ci_low,d.ci_high-d['mean']],fmt='o',color=TEAL,capsize=3)
    ax.set_yticks(y,d.label);ax.invert_yaxis();ax.set(xlabel='Post − pre (original 0–10 item)',title='Within-person item changes')
    save(fig,'f1_h_item_changes',f'Paired post-minus-pre differences use the same participants at both occasions. Bars are percentile 95% intervals from {bootstrap:,} participant resamples. These are unadjusted descriptive intervals for six separate items; they are not simultaneous intervals. Raw supervision direction is retained.')

    trans=[];colors=[NAVY,GOLD,TEAL];bin_labels=['0–3','4–6','7–10']
    for ni,(fam,item,label) in enumerate(ITEMS,1):
        pre=p[f'{fam}_pre_{item}'];post=p[f'{fam}_post_{item}'];ok=pre.notna()&post.notna()
        a=np.digitize(pre[ok],[3.5,6.5]);b=np.digitize(post[ok],[3.5,6.5]);counts=np.zeros((3,3),int)
        for aa,bb in zip(a,b):counts[aa,bb]+=1
        for aa in range(3):
            for bb in range(3):trans.append({'family':fam,'item':item,'pre_bin':bin_labels[aa],'post_bin':bin_labels[bb],'n':int(counts[aa,bb]),'fraction_all_pairs':counts[aa,bb]/ok.sum(),'n_pairs':int(ok.sum())})
        total=counts.sum();scale=.80/total;gap=.045
        def starts(sizes):return np.r_[.035,.035+np.cumsum(sizes[:-1])*scale+np.arange(1,3)*gap]
        left=starts(counts.sum(axis=1));right=starts(counts.sum(axis=0));lc=left.copy();rc=right.copy()
        fig,ax=plt.subplots(figsize=(4.6,3.6));ax.set(xlim=(-.03,1.03),ylim=(-.03,1.08));ax.axis('off')
        for aa in range(3):
            for bb in range(3):
                h=counts[aa,bb]*scale
                if h==0:continue
                vertices=[(.18,lc[aa]),(.44,lc[aa]),(.56,rc[bb]),(.82,rc[bb]),(.82,rc[bb]+h),(.56,rc[bb]+h),(.44,lc[aa]+h),(.18,lc[aa]+h),(.18,lc[aa])]
                codes=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY]
                ax.add_patch(PathPatch(MPath(vertices,codes),facecolor=colors[aa],alpha=.30,edgecolor='none'));lc[aa]+=h;rc[bb]+=h
        for j in range(3):
            for x,ys,n,align,tx in [(.14,left[j],counts[j].sum(),'right',.12),(.82,right[j],counts[:,j].sum(),'left',.88)]:
                hh=n*scale;ax.add_patch(Rectangle((x,ys),.04,hh,facecolor=colors[j],edgecolor='none'))
                ax.text(tx,ys+hh/2,f'{bin_labels[j]}\n{n/total:.0%}',ha=align,va='center',fontsize=8)
        ax.text(.16,1.02,'Pre',ha='center',fontsize=10);ax.text(.84,1.02,'Post',ha='center',fontsize=10)
        ax.set_title(label,loc='left',pad=12)
        footer_y=-.10 if item=='monitor_raw' else -.015
        if item=='monitor_raw':ax.set_ylim(-.15,1.08)
        ax.text(.5,footer_y,f'n = {total:,} paired responses · fixed score bands',ha='center',fontsize=8)
        save(fig,f'f1_i{ni}_{fam}_{item}_transitions','Flows show paired original-item score transitions between prespecified descriptive bands 0–3, 4–6, and 7–10. These integer cutpoints are arbitrary summaries, not estimated thresholds, latent states, or evidence for discrete psychological groups. Width is participant count; zero remains valid. Supervision retains its original direction.')
    pd.DataFrame(trans).to_csv(tab/'d2_item_transitions_fixed_bands.csv',index=False)

    fields=['age','driving_exp_years','risk_person_block_balanced_mean']+[f'{fam}_pre_{item}' for fam,item,_ in ITEMS]
    labels=['Age','Experience','Mean risk','Pre: overall trust','Pre: ACC trust','Pre: LKS trust','Pre: delegate','Pre: supervision','Pre: non-driving']
    rows=[];rho=np.eye(len(fields));ns=np.zeros((len(fields),len(fields)),int)
    for i in range(len(fields)):
        ns[i,i]=p[fields[i]].notna().sum()
        for j in range(i+1,len(fields)):
            a=p[[fields[i],fields[j]]].dropna();r,pp=spearmanr(a.iloc[:,0],a.iloc[:,1]);rho[i,j]=rho[j,i]=r;ns[i,j]=ns[j,i]=len(a)
            rows.append({'variable_a':fields[i],'variable_b':fields[j],'rho':r,'p':pp,'n_participants':len(a)})
    cr=pd.DataFrame(rows);cr['q_bh']=bh(cr.p);cr.to_csv(tab/'d2_spearman_associations.csv',index=False)
    qmat=np.ones_like(rho)
    for row in cr.itertuples():i=fields.index(row.variable_a);j=fields.index(row.variable_b);qmat[i,j]=qmat[j,i]=row.q_bh
    fig,ax=plt.subplots(figsize=(7.0,6.0));img=ax.imshow(rho,vmin=-1,vmax=1,cmap='RdBu_r')
    for i in range(len(fields)):
        for j in range(len(fields)):
            if j<=i:ax.text(j,i,f'{rho[i,j]:.2f}'+('*' if i!=j and qmat[i,j]<.05 else ''),ha='center',va='center',fontsize=8,color='white' if abs(rho[i,j])>.55 else NAVY)
            else:ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor='white',edgecolor='white'))
    ax.set_xticks(range(len(fields)),labels,rotation=55,ha='right',fontsize=8);ax.set_yticks(range(len(fields)),labels,fontsize=8)
    ax.set_title('Associations at the participant level',loc='left',pad=12);fig.colorbar(img,ax=ax,fraction=.045,pad=.03,label='Spearman ρ')
    ax.text(0,-.37,f'Pairwise n = {cr.n_participants.min():,}–{cr.n_participants.max():,} · * BH q < .05 across {len(cr)} pairs',transform=ax.transAxes,fontsize=8)
    save(fig,'f1_j_spearman_associations','Spearman rank associations use one row per participant and pairwise complete observations. Risk is participant block-balanced mean of valid event means. Asterisks indicate Benjamini–Hochberg q<.05 across 36 unique pairs. Exact pairwise n, p, and q are supplied. Associations describe data coherence and covariation, not psychometric validity, causality, or independent replication.')

    # Country profiles: participants first, then countries. All 21 positions retained.
    x=w.groupby(['participant_id','scenario','window_index']).risk.mean().unstack(['scenario','window_index'])
    x.columns=[f'{a}_W{b}' for a,b in x.columns];positions=list(x.columns);x=x.join(p.set_index('participant_id').country)
    countries=sorted(x.country.unique());arrays=[x.loc[x.country.eq(c),positions].to_numpy(float) for c in countries]
    n=np.array([len(a) for a in arrays]);means=np.array([np.nanmean(a,axis=0) for a in arrays]);assert np.isfinite(means).all()
    centered=means-means.mean(axis=1,keepdims=True)
    z=linkage(centered,method='ward',optimal_ordering=True);tree=to_tree(z);leaf_order=leaves_list(z).tolist()
    def clades(node, dest):
        if node.is_leaf():return frozenset([node.id])
        members=clades(node.left,dest)|clades(node.right,dest);dest[node.id]=members;return members
    base={};clades(tree,base);hits={k:0 for k in base};accepted=0;rejected=0
    while accepted<tree_bootstrap:
        if accepted+rejected>tree_bootstrap*10:raise RuntimeError('Too many bootstrap profiles with undefined positions; inspect missingness before plotting.')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            bm=np.array([np.nanmean(a[rng.integers(0,len(a),size=len(a))],axis=0) for a in arrays])
        if not np.isfinite(bm).all():rejected+=1;continue
        bm=bm-bm.mean(axis=1,keepdims=True);bt=to_tree(linkage(bm,method='ward'));bc={};clades(bt,bc);sets=set(bc.values())
        for k,members in base.items():hits[k]+=members in sets
        accepted+=1
    support={k:hits[k]/accepted for k in base};profile=[]
    for ci,c in enumerate(countries):
        for j,pos in enumerate(positions):profile.append({'country':c,'position':pos,'n_participants_country':int(n[ci]),'n_participants_position':int(np.isfinite(arrays[ci][:,j]).sum()),'mean_risk':means[ci,j],'country_centered_mean':centered[ci,j],'small_n_flag':bool(n[ci]<30)})
    pd.DataFrame(profile).to_csv(tab/'d2_country_position_profiles.csv',index=False)
    stability=[]
    for k,members in base.items():
        ids=sorted(members);minimum=int(n[ids].min());estimable=minimum>1
        stability.append({'node_id':k,'member_countries':' | '.join(countries[i] for i in ids),'n_countries':len(ids),'minimum_country_n':minimum,'any_small_n_country':bool((n[ids]<30).any()),'bootstrap_clade_recovery':support[k] if estimable else np.nan,'conditional_recovery_including_degenerate_singletons':support[k],'uncertainty_estimable_for_all_members':estimable,'bootstrap_replicates':accepted,'bootstrap_incomplete_profile_rejections':rejected})
    sd=pd.DataFrame(stability);sd.to_csv(tab/'d2_country_tree_stability.csv',index=False)
    fig=plt.figure(figsize=(9.3,9.3));ax=fig.add_subplot(111,projection='polar');ax.set_theta_offset(np.pi/2);ax.set_theta_direction(-1);ax.set_ylim(0,1.45);ax.set_axis_off()
    theta={i:(j+.5)*2*np.pi/len(countries) for j,i in enumerate(leaf_order)};maxdist=tree.dist
    def draw(node):
        if node.is_leaf():return theta[node.id],1.0
        tl,rl=draw(node.left);tr,rr=draw(node.right);t=(tl+tr)/2;r=.12+.88*(1-node.dist/maxdist)
        ax.plot([tl,tl],[r,rl],color='#80909B',lw=1);ax.plot([tr,tr],[r,rr],color='#80909B',lw=1);ts=np.linspace(tl,tr,60);ax.plot(ts,np.full_like(ts,r),color='#80909B',lw=1)
        members=list(base[node.id]);estimable=(n[members]>1).all()
        if node.id!=tree.id and estimable and support[node.id]>=.50:
            ax.text(t,r,f'{support[node.id]:.0%}',fontsize=6.5,ha='center',va='center',bbox={'facecolor':'white','edgecolor':'none','pad':1})
        return t,r
    draw(tree)
    for i,c in enumerate(countries):
        t=theta[i];degrees=np.degrees(t);rotation=90-degrees;align='left'
        if degrees>180:rotation+=180;align='right'
        color=TEAL if n[i]>=30 else (RUST if n[i]==1 else '#7D8790')
        ax.plot(t,1.01,'o',ms=3+1.4*np.log10(n[i]),color=color)
        label=f'{SHORT.get(c,c)} · {n[i]}' + (' †' if n[i]<30 else '')
        ax.text(t,1.045,label,rotation=rotation,rotation_mode='anchor',ha=align,va='center',fontsize=8,color=color)
    fig.suptitle('Exploratory similarity of country risk profiles',x=.5,y=.94,fontsize=13)
    fig.text(.5,.08,'21 discrete questionnaire positions · country-centred means · Ward clustering\n† n < 30; rust n = 1 has no estimable within-country uncertainty\nNode labels: bootstrap clade recovery ≥50% where every country has n > 1',ha='center',fontsize=8)
    save(fig,'f1_k_country_profile_radial_tree',f'Within each participant, valid event scores are averaged at each of the 21 block/window positions. Country means are then centred across positions without scaling. Ward clustering uses Euclidean distances between these centred country profiles. All 29 countries remain as leaves; the tree is exploratory and does not estimate culture types. Participant resampling occurs within each country with replacement, preserving multiplicity ({accepted} accepted replicates; {rejected} replicates rejected because a profile position was undefined). Branch recovery is exact descendant-set recovery, not a probability that a grouping is true. Clades containing an n=1 country have no reported inferential stability because resampling that country is degenerate. n<30 is flagged throughout; retained conditional recovery is supplied only for transparency. No singleton country supports a population-level generalisation.')

    # Simple sensitivity summaries do not replace the all-participant main analysis.
    sensitivity=[]
    for label,ids in [('all_completed_licensed',set(p.participant_id)),('duration_qc_pass',set(p.loc[p.duration_qc_pass,'participant_id']))]:
        pp=p[p.participant_id.isin(ids)];ee=e[e.participant_id.isin(ids)];ww=w[w.participant_id.isin(ids)]
        for block,g in ee.groupby('scenario'):
            individual=g.groupby('participant_id').risk_event_mean.mean();sensitivity.append({'cohort':label,'variable':f'{block}_risk_event_mean','n':individual.notna().sum(),'mean':individual.mean()})
        for fam,item,_ in ITEMS:
            delta=pp[f'{fam}_post_{item}']-pp[f'{fam}_pre_{item}'];sensitivity.append({'cohort':label,'variable':f'{fam}_{item}_delta','n':delta.notna().sum(),'mean':delta.mean()})
        sensitivity.append({'cohort':label,'variable':'no_operation_fraction','n':len(ww),'mean':ww.risk_raw.eq(0).mean()})
    pd.DataFrame(sensitivity).to_csv(tab/'d2_duration_selection_sensitivity.csv',index=False)
    pd.DataFrame(captions).to_csv(rep/'d2_figure_captions.csv',index=False)
    manifest={'input_sha256':inputs,'script_sha256':digest(Path(__file__)),'seed':seed,'bootstrap_replicates':bootstrap,'tree_bootstrap_replicates':tree_bootstrap,'tree_rejected_replicates':rejected,'figures':files,'all_country_n':int(p.country.nunique()),'n_participants':len(p),'n_nonzero_risk_ratings':int(w.risk.notna().sum()),'zero_is_missing_only_for_video_risk':True}
    (rep/'d2_descriptive_manifest.json').write_text(json.dumps(manifest,indent=2))
    methods='''# Descriptive analysis and figure notes

Primary sample: all 2,164 completed, licensed respondents in all 29 reported countries. Duration and country-size thresholds do not select the main sample. There are 179,966 usable nonzero video risk ratings; 1,810 administered slots coded zero are marked no operation. Trust/acceptance 0 remains valid.

Original demographic answers are retained rather than silently corrected. Seven respondents have age minus driving experience below 16, including an internally implausible value of zero. These are source-quality flags, not an assessment of country-specific licensing rules and not a new sample exclusion. Gender is missing for 23 respondents, who remain in a separate descriptive group. The aggregate checks are in d2_demographic_quality_flags.csv.

Uncertainty intervals resample whole participants with replacement and preserve repeated draws. Risk summaries first aggregate repeated observations within participants. Pre/post differences are paired; each bootstrap draw includes both occasions from the same participant. Pointwise 95% intervals are descriptive and not simultaneous. Gender panels are unadjusted descriptions.

The original six item questions are retained. Need for supervision is shown in its raw direction. Sankey score bands (0–3, 4–6, 7–10) are fixed descriptive summaries; no latent classes or transitions between psychological states are inferred. Correlations use one row per participant; BH correction applies to all 36 unique pairs. Association is not a validation test.

The country tree is an exploratory comparison of centred 21-position risk profiles. Countries receive equal geometric weight, regardless of n, and therefore small-sample leaves can have large estimation error. All countries remain visible. Bootstrap clade recovery is an empirical resampling diagnostic, not evidence of cultural categories; exact recovery depends on the chosen distance/linkage. Clades involving singleton countries have no estimable uncertainty. Country n and position-specific n are supplied. The current sample has no undefined country-position means, so no imputation was used in the observed tree. Bootstrap draws with an undefined country-position mean are rejected and counted, which conditions the diagnostic on an estimable profile.

Risk position is a discrete questionnaire-window index. It is neither reconstructed elapsed seconds nor cross-block presentation order. Physical scenario mapping remains unresolved. Country profiles may reflect participant composition and the assigned event mix, and do not isolate causal country effects.
'''
    methods+='\n## Panels\n\n'+'\n\n'.join(f"**{c['figure']}**: {c['caption']}" for c in captions)
    (rep/'d2_descriptive_methods.md').write_text(methods)
    assert all(digest(data/f'{name}.parquet')==h for name,h in inputs.items())
    print(json.dumps({'panels':len(captions),'figure_files':len(files),'countries':len(countries),'tree_bootstrap_accepted':accepted,'tree_bootstrap_rejected':rejected,'inputs_unchanged':True},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--bootstrap',type=int,default=1000);parser.add_argument('--tree-bootstrap',type=int,default=500);parser.add_argument('--seed',type=int,default=20261001)
    args=parser.parse_args();main(args.root,args.bootstrap,args.tree_bootstrap,args.seed)
