"""Full design-factor summaries and additive within-person sensitivity.

Implements the study design-factor specification. The historical frozen-plan and
old-output comparisons remain in the private project; this portable version
checks current inputs before and after and retains mathematical WLS checks. No raw workbook/MAT/model edits,
no selection by significance, no Spearman encoding of nominal behaviours.
Run with BLAS threads limited to2. Requires numpy,pandas,pyarrow.
"""
from settings import WORKSPACE as ROOT, DATA_DIR, QUESTIONNAIRE, REPO, relative_label
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
import numpy as np
import pandas as pd
from design_spec import SPECS,BLOCKS

SEED=20261007;B=2000
FIELD={'design_speed':'design_speed_kmh','design_braking':'design_braking_m_s2','design_distance':'design_distance_m','lateral_behaviour':'lateral_behaviour','acc_style':'acc_style'}
UNIT={'design_speed':'km/h','design_braking':'m/s²','design_distance':'m','lateral_behaviour':'category','acc_style':'category'}
LABEL={'1m/s':'Normal, 1 m/s','3m/s':'Normal, 3 m/s','Fragmented':'Fragmented','Abortion':'Aborted','cautious':'Cautious','mild':'Mild','aggressive':'Aggressive'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def label(f,v):return LABEL.get(v,str(v)+' '+UNIT[f])
def main(root):
 dest=root/'outputs/parameters';tables=dest/'tables';reports=dest/'reports';data=dest/'data'
 for directory in [tables,reports,data]:directory.mkdir(parents=True,exist_ok=True)
 paths={'design':REPO/'data/event_design.csv','windows':DATA_DIR/'windows.parquet','participants':DATA_DIR/'participants.parquet'}
 inputs={k:{'path':relative_label(p,root),'sha256':sha(p)} for k,p in paths.items()}
 people=pd.read_parquet(paths['participants']);ids=np.sort(people.participant_id.to_numpy());assert len(ids)==2164 and len(set(ids))==2164 and people.country.nunique()==29
 w=pd.read_parquet(paths['windows']);assert len(w)==181776 and w.risk.notna().sum()==179966 and w.risk_raw.eq(0).sum()==1810
 design=pd.read_csv(paths['design']);w=w.merge(design,on=['event_key','scenario'],how='left',validate='many_to_one',suffixes=('','_design'));assert w.source_event_name.notna().all()
 weights=np.random.default_rng(SEED).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=B).astype(float)
 weight_sha=hashlib.sha256(weights.astype(np.int16).tobytes()).hexdigest()
 curves=[];paired=[];adjusted=[];model_rows=[];math_checks=[];replicas={'profiles':[],'paired':[],'adjusted':[]};rep_columns=[]
 def add_replicas(kind,array,keys):
  start=sum(x.shape[1] for x in replicas[kind]);replicas[kind].append(array)
  for j,key in enumerate(keys):rep_columns.append(dict(kind=kind,column=start+j,key=key))
 def summarize(a):
  available=np.isfinite(a);den=weights@available.astype(float);rep=np.divide(weights@np.nan_to_num(a,nan=0.),den,out=np.full((B,a.shape[1]),np.nan),where=den>0)
  return np.nanmean(a,axis=0),np.nanquantile(rep,[.025,.975],axis=0),rep
 def profiles_and_pairs(code,variant,factors,drop_conflicts=False):
  sub=w[w.display_scenario.eq(code)].copy()
  if drop_conflicts:sub=sub[~sub.lc_distance_conflict]
  k=6 if code=='LC' else 5
  for factor in factors:
   col=FIELD[factor];levels=SPECS[code][factor];means=sub.groupby(['participant_id',col,'window_index']).risk.mean();matrices=[]
   for lev in levels:matrices.append(means.xs(lev,level=col).unstack('window_index').reindex(index=ids,columns=range(1,k+1)).to_numpy())
   combined=np.concatenate(matrices,axis=1);point,interval,rep=summarize(combined);keys=[]
   for li,lev in enumerate(levels):
    for clip in range(1,k+1):
     index=li*k+clip-1;cell=sub[sub[col].eq(lev)&sub.window_index.eq(clip)];key=f'{variant}|{code}|{factor}|{lev}|{clip}';keys.append(key)
     curves.append(dict(analysis_variant=variant,scenario=BLOCKS[code],display_scenario=code,factor=factor,level=lev,level_label=label(factor,lev),level_order=li+1,unit=UNIT[factor],distance_source_label=cell.distance_source_label.iloc[0],clip=clip,n_participants=int(np.isfinite(combined[:,index]).sum()),n_assigned_participants=cell.participant_id.nunique(),n_assigned_event_ratings=len(cell),n_valid_ratings=cell.risk.notna().sum(),n_no_operation=cell.risk_raw.eq(0).sum(),n_distinct_events=cell.event_key.nunique(),mean=point[index],participant_sd=np.nanstd(combined[:,index],ddof=1),ci_low=interval[0,index],ci_high=interval[1,index],bootstrap_draws=B,bootstrap_valid=int(np.isfinite(rep[:,index]).sum()),row_key=key))
   add_replicas('profiles',rep,keys)
   comparisons=range(1,len(levels)) if factor=='lateral_behaviour' else [len(levels)-1]
   for j in comparisons:
    delta=matrices[j]-matrices[0];point,interval,rep=summarize(delta);keys=[]
    for clip in range(1,k+1):
     key=f'{variant}|{code}|{factor}|{levels[j]}-{levels[0]}|{clip}';keys.append(key)
     paired.append(dict(analysis_variant=variant,scenario=BLOCKS[code],display_scenario=code,factor=factor,clip=clip,reference_level=levels[0],comparison_level=levels[j],reference_label=label(factor,levels[0]),comparison_label=label(factor,levels[j]),unit=UNIT[factor],n_paired_participants=int(np.isfinite(delta[:,clip-1]).sum()),mean_difference=point[clip-1],ci_low=interval[0,clip-1],ci_high=interval[1,clip-1],bootstrap_draws=B,bootstrap_valid=int(np.isfinite(rep[:,clip-1]).sum()),contrast_definition='Comparison minus reference, same person; other factors marginal, not matched',row_key=key))
    add_replicas('paired',rep,keys)
 def within_stats(sub,spec):
  keep=sub[sub.risk.notna()].copy();pidpos=pd.Index(ids).get_indexer(keep.participant_id);assert (pidpos>=0).all();n=np.bincount(pidpos,minlength=len(ids));y=keep.risk.to_numpy(float);cols=[];meta=[]
  for factor,levels in spec.items():
   for lev in levels[1:]:cols.append(keep[FIELD[factor]].eq(lev).to_numpy(float));meta.append((factor,levels[0],lev))
  x=np.column_stack(cols);q=x.shape[1];sx=np.zeros((len(ids),q));np.add.at(sx,pidpos,x);sy=np.bincount(pidpos,weights=y,minlength=len(ids));xc=x-sx[pidpos]/n[pidpos,None];yc=y-sy[pidpos]/n[pidpos]
  xx=np.zeros((len(ids),q,q));xy=np.zeros((len(ids),q));np.add.at(xx,pidpos,np.einsum('ni,nj->nij',xc,xc)/n[pidpos,None,None]);np.add.at(xy,pidpos,xc*yc[:,None]/n[pidpos,None]);return keep,n,xx,xy,meta,x,y,pidpos
 def adjusted_models(code,variant,drop_conflicts=False):
  sub=w[w.display_scenario.eq(code)].copy();spec={k:list(v) for k,v in SPECS[code].items()}
  if drop_conflicts:sub=sub[~sub.lc_distance_conflict];spec['lateral_behaviour']=spec['lateral_behaviour'][:-1]
  for clip in range(1,7 if code=='LC' else 6):
   cell=sub[sub.window_index.eq(clip)];keep,n,xx,xy,meta,x,y,pidpos=within_stats(cell,spec);q=len(meta);a=xx.sum(axis=0);b=xy.sum(axis=0);rank=np.linalg.matrix_rank(a);point=np.linalg.solve(a,b) if rank==q else np.full(q,np.nan)
   boot_a=np.tensordot(weights,xx,axes=(1,0));boot_b=weights@xy;boot_rank=np.linalg.matrix_rank(boot_a);ok=boot_rank==q;rep=np.full((B,q),np.nan)
   if ok.any():rep[ok]=np.linalg.solve(boot_a[ok],boot_b[ok,:,None])[...,0]
   interval=np.nanquantile(rep,[.025,.975],axis=0) if ok.any() else np.full((2,q),np.nan);key=f'{variant}|{code}|{clip}'
   contributing=(np.trace(xx,axis1=1,axis2=2)>1e-12);model_rows.append(dict(analysis_variant=variant,scenario=BLOCKS[code],display_scenario=code,clip=clip,n_valid_ratings=len(keep),n_participants_with_rating=int((n>0).sum()),n_participants_with_two_or_more=int((n>=2).sum()),n_participants_with_within_design_variation=int(contributing.sum()),n_parameters=q,design_rank=int(rank),estimable=rank==q,bootstrap_valid=int(ok.sum()),bootstrap_rank_min=int(boot_rank.min()),bootstrap_draws=B))
   keys=[]
   for j,(factor,ref,lev) in enumerate(meta):
    rowkey=f'{key}|{factor}|{lev}-{ref}';keys.append(rowkey);adjusted.append(dict(analysis_variant=variant,scenario=BLOCKS[code],display_scenario=code,clip=clip,factor=factor,reference_level=ref,comparison_level=lev,reference_label=label(factor,ref),comparison_label=label(factor,lev),unit=UNIT[factor],estimate=point[j],ci_low=interval[0,j],ci_high=interval[1,j],n_participants_with_rating=int((n>0).sum()),n_participants_with_within_design_variation=int(contributing.sum()),n_valid_ratings=len(keep),n_parameters=q,design_rank=int(rank),status='estimated' if rank==q else 'rank_deficient',bootstrap_valid=int(ok.sum()),bootstrap_draws=B,coefficient_scope='Additive weighted participant fixed-effects sensitivity; categorical design main effects; no interaction',row_key=rowkey))
   add_replicas('adjusted',rep,keys)
   if clip==1:
    # Independent direct dummy-intercept weighted least squares on a small,
    # deterministic subset; does not use within-system cross-products.
    selected=np.unique(keep.participant_id)[:60];sm=cell[cell.participant_id.isin(selected)];sk,sn,sxx,sxy,smeta,sx,sy,sp=within_stats(sm,spec);sr=np.linalg.matrix_rank(sxx.sum(axis=0));active=np.flatnonzero(sn>0);dummy=(sp[:,None]==active[None,:]).astype(float);full=np.column_stack([dummy,sx]);sw=np.sqrt(1/sn[sp]);direct=np.linalg.lstsq(full*sw[:,None],sy*sw,rcond=None)[0][-q:];within=np.linalg.solve(sxx.sum(axis=0),sxy.sum(axis=0)) if sr==q else np.full(q,np.nan);err=float(np.max(np.abs(direct-within))) if sr==q else None
    assert err is None or err<1e-10
    math_checks.append(dict(analysis_variant=variant,display_scenario=code,clip=1,participants=len(active),observations=len(sk),within_rank=int(sr),coefficient_count=q,maximum_absolute_difference=err,comparison='Explicit participant dummy-intercept WLS versus within transformation; each participant total weight1'))
 for code in ['LC','SVM','HB','MB']:profiles_and_pairs(code,'assigned_design_primary',SPECS[code]);adjusted_models(code,'assigned_design_primary')
 profiles_and_pairs('LC','lc_exclude_distance_conflicts',['design_distance'],True);adjusted_models('LC','lc_exclude_distance_conflicts',True)
 c=pd.DataFrame(curves);d=pd.DataFrame(paired);f=pd.DataFrame(adjusted);models=pd.DataFrame(model_rows);assert len(c)==201 and len(d)==81 and len(f)==156 and len(models)==27
 assert c[c.analysis_variant.eq('assigned_design_primary')].shape[0]==189 and d[d.analysis_variant.eq('assigned_design_primary')].shape[0]==75 and f[f.analysis_variant.eq('assigned_design_primary')].shape[0]==126
 assert c.n_valid_ratings.add(c.n_no_operation).eq(c.n_assigned_event_ratings).all() and c.n_participants.le(c.n_assigned_participants).all()
 assert np.isfinite(c[['mean','participant_sd','ci_low','ci_high']]).all().all() and np.isfinite(d[['mean_difference','ci_low','ci_high']]).all().all()
 # Planned simultaneous sensitivity across all75 primary contrasts, never LC diagnostic.
 rep_paired=np.concatenate(replicas['paired'],axis=1);mask=d.analysis_variant.eq('assigned_design_primary').to_numpy();primary_rep=rep_paired[:,mask];point=d.loc[mask,'mean_difference'].to_numpy();se=np.nanstd(primary_rep,axis=0,ddof=1);constant=(se==0)&np.all(np.isclose(primary_rep,point[None,:],atol=1e-12,rtol=0),axis=0);estimable=np.isfinite(point)&np.isfinite(se)&((se>0)|constant);complete=np.isfinite(primary_rep[:,estimable]).all(axis=1)
 standardized=np.divide(primary_rep[complete][:,estimable]-point[estimable],se[estimable],out=np.zeros((complete.sum(),estimable.sum())),where=se[estimable]>0);critical=float(np.quantile(np.max(np.abs(standardized),axis=1),.95)) if complete.any() and estimable.any() else np.nan
 pi=np.flatnonzero(mask);d.loc[mask,'simultaneous_family_planned']=75;d.loc[mask,'simultaneous_family_estimable']=int(estimable.sum());d.loc[mask,'simultaneous_complete_draws']=int(complete.sum());d.loc[mask,'simultaneous_critical']=critical;d.loc[mask,'bootstrap_se']=se;d.loc[pi[estimable],'simultaneous_ci_low']=point[estimable]-critical*se[estimable];d.loc[pi[estimable],'simultaneous_ci_high']=point[estimable]+critical*se[estimable]
 outputs={'controlled_parameter_clip_profiles.csv':c,'controlled_parameter_paired_contrasts.csv':d,'controlled_parameter_adjusted_sensitivity.csv':f,'controlled_parameter_fe_model_coverage.csv':models}
 for name,frame in outputs.items():frame.to_csv(tables/name,index=False)
 pd.DataFrame(rep_columns).to_csv(reports/'controlled_parameter_bootstrap_columns.csv',index=False)
 np.savez_compressed(data/'controlled_parameter_bootstrap.npz',**{k:np.concatenate(v,axis=1) for k,v in replicas.items()})
 for k,p in paths.items():assert sha(p)==inputs[k]['sha256']
 result={'status':'pass','created_utc':datetime.now(timezone.utc).isoformat(),'source_hashes':inputs,'analysis_inputs_unchanged':True,'participant_n':len(ids),'country_n':29,'draws':B,'seed':SEED,'shared_weight_matrix_sha256':weight_sha,'primary_profile_rows':189,'primary_paired_contrast_rows':75,'primary_FE_models':21,'primary_FE_coefficients':126,'lc_exclusion_profile_rows':12,'lc_exclusion_paired_rows':6,'lc_exclusion_FE_models':6,'lc_exclusion_FE_coefficients':30,'simultaneous_primary_family':{'planned':75,'estimable':int(estimable.sum()),'joint_complete_draws':int(complete.sum()),'critical':critical,'method':'Fixed bootstrap-SE maximum-standardised-deviation sensitivity, not nested bootstrap-t; primary family only.'},'FE_model_rank_deficient':int((models.design_rank<models.n_parameters).sum()),'FE_bootstrap_valid_range':[int(models.bootstrap_valid.min()),int(models.bootstrap_valid.max())],'direct_dummy_WLS_independent_checks':math_checks,'interpretation':'Primary curves/paired contrasts are marginal over the other design factors. FE is an additive conditional-mean sensitivity, not causal isolation or population stimulus generalisation. No new p values. Pointwise interval exclusions across75contrasts must be considered with the simultaneous sensitivity.','output_hashes':{str(p.relative_to(root)):sha(p) for p in [*[tables/n for n in outputs],reports/'controlled_parameter_bootstrap_columns.csv',data/'controlled_parameter_bootstrap.npz']},'script_sha256':sha(Path(__file__))}
 (reports/'controlled_parameter_analysis_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','primary_profile_rows','primary_paired_contrast_rows','primary_FE_coefficients','simultaneous_primary_family','FE_model_rank_deficient','FE_bootstrap_valid_range']},ensure_ascii=False,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project-root',type=Path,default=ROOT);main(ap.parse_args().project_root.resolve())
