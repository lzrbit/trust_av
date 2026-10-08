"""NHB round 2 source build: risk zero means no operation; trust zero is valid.

Primary cohort: all completed, licensed respondents, all reported countries.
Duration quantiles are sensitivity flags only. No source files are modified.
"""
from __future__ import annotations
from settings import WORKSPACE as ROOT, DATA_DIR, QUESTIONNAIRE, REPO, relative_label
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl


BLOCKS={'LC':'B1','SVM':'B2','HB':'B3','MB':'B4'}
N_WINDOWS={'B1':6,'B2':5,'B3':5,'B4':5}
LABELS={'B1':'LC','B2':'SVM','B3':'HB','B4':'MB'}

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def clean(series):
    return series.astype('string').str.strip().replace({'':pd.NA,'nan':pd.NA,'None':pd.NA})

def save(df,path):
    temporary=path.with_suffix('.parquet.tmp');df.to_parquet(temporary,index=False);temporary.replace(path)

def main(root):
    out=root/'outputs/core';data=DATA_DIR;tables=out/'tables';reports=out/'reports'
    for path in [data,tables,reports]:path.mkdir(parents=True,exist_ok=True)
    raw_files={'questionnaire.xlsx': QUESTIONNAIRE}
    before={name:digest(path) for name,path in raw_files.items()}
    xlsx=QUESTIONNAIRE
    wb=openpyxl.load_workbook(xlsx,read_only=True,data_only=True);sheet=wb['Publish-OnlinePerceivedSafetyWi']
    it=sheet.iter_rows(values_only=True);head,labels,imports=next(it),next(it),next(it)
    raw=pd.DataFrame(list(it),columns=range(1,len(head)+1));wb.close()
    num=lambda c:pd.to_numeric(raw[c],errors='coerce')
    p=pd.DataFrame({'participant_id':np.arange(4,len(raw)+4),'source_excel_row':np.arange(4,len(raw)+4)})
    p['source_row_id']=p.source_excel_row.map(lambda v:f'xlsx_row_{v:04d}')
    p['progress']=num(5);p['license_yes']=clean(raw[20]).str.lower().eq('yes').fillna(False)
    p['completed_licensed']=p.progress.eq(100)&p.license_yes
    p['country_raw']=clean(raw[22]);p['country_missing']=p.country_raw.isna();p['country']=p.country_raw.fillna('Unknown')
    p['birth_year']=num(23);p['license_year']=num(28)
    p['age']=num(24).fillna(2023-p.birth_year);p['driving_exp_years']=num(29).fillna(2023-p.license_year)
    p['age_source']=np.where(num(24).notna(),'xlsx_column24','2023_minus_birth_year')
    p['experience_source']=np.where(num(29).notna(),'xlsx_column29','2023_minus_license_year')
    p['gender']=clean(raw[25]);p['education']=clean(raw[26]);p['duration_seconds']=num(6)
    p['recorded_year']=pd.to_datetime(raw[8],errors='coerce').dt.year.astype('Int64')
    qlo,qhi=p.loc[p.completed_licensed,'duration_seconds'].quantile([.05,.99])
    p['duration_qc_pass']=p.duration_seconds.between(qlo,qhi)
    p['qc_duration']=p['duration_qc_pass']
    p['duration_qc_role']='sensitivity_only_not_primary_exclusion'
    extras={30:'own_car',31:'drive_freq_12m',32:'km_life',33:'km_12m',34:'drive_type',35:'drive_style',36:'know_ad',37:'acc_years',38:'acc_dist',39:'lks_years',40:'lks_dist',41:'acclks_years',42:'acclks_dist'}
    codebook=[]
    for c,name in extras.items():
        p[name]=clean(raw[c]);codebook.append({'field':name,'source_column':c,'question':labels[c-1],'zero_rule':'source categorical response retained'})
    for when,cols in [('pre',range(43,49)),('post',range(1062,1068))]:
        names=[f'trust_{when}_overall',f'trust_{when}_acc',f'trust_{when}_lks',f'accept_{when}_delegate',f'accept_{when}_monitor_raw',f'accept_{when}_distract']
        for c,name in zip(cols,names):
            p[name]=num(c);codebook.append({'field':name,'source_column':c,'question':labels[c-1],'zero_rule':'0 is a valid 0–10 response; explicitly confirmed by user'})
        p[f'accept_{when}_monitor_reversed']=10-p[f'accept_{when}_monitor_raw']
        p[f'accept_{when}_monitor']=p[f'accept_{when}_monitor_reversed']
        p[f'trust_{when}_mean3']=p[[f'trust_{when}_{s}' for s in ['overall','acc','lks']]].mean(axis=1,skipna=False)
        p[f'trust_{when}_func']=p[f'trust_{when}_mean3']
    for sub in ['overall','acc','lks','mean3']:p[f'trust_delta_{sub}']=p[f'trust_post_{sub}']-p[f'trust_pre_{sub}']
    for sub in ['delegate','monitor_raw','monitor_reversed','distract']:p[f'accept_delta_{sub}']=p[f'accept_post_{sub}']-p[f'accept_pre_{sub}']
    source_p=p.copy();p=p[p.completed_licensed].copy().reset_index(drop=True)
    positions=p.source_excel_row.to_numpy()-4
    colmap_path=REPO/'config/risk_column_mapping.json';cm=pd.DataFrame(json.loads(colmap_path.read_text()))
    assert len(cm)==549 and not cm.col_1indexed.duplicated().any()
    assert all(str(head[c-1])==q for c,q in zip(cm.col_1indexed,cm.qid))
    cm['questionnaire_block']=cm.scenario.map(BLOCKS)
    events=[];windows=[];assignments=[];design=[]
    for (block,eid),g in cm.groupby(['questionnaire_block','event_id'],sort=True):
        g=g.sort_values('window_id');cols=g.col_1indexed.tolist();a=raw.loc[:,cols].apply(pd.to_numeric,errors='coerce').to_numpy()[positions]
        assigned=np.isfinite(a).any(axis=1);valid=np.isfinite(a)&(a>0)&(a<=10);k=a.shape[1]
        assignments.append(pd.DataFrame({'participant_id':p.participant_id,'questionnaire_block':block,'questionnaire_event_id':eid,'event_key':f'{block}_E{eid:02d}','event_assigned':assigned,'assignment_missing_reason':np.where(assigned,'','not_assigned_structural')}))
        ids=p.loc[assigned,'participant_id'].to_numpy();av=a[assigned];vv=valid[assigned];nvalid=vv.sum(axis=1)
        rec=pd.DataFrame({'participant_id':ids,'questionnaire_block':block,'questionnaire_event_id':eid,'scenario':block,'event_id':eid,'event_key':f'{block}_E{eid:02d}'})
        rec['event_row_id']=rec.participant_id.astype(str)+'_'+rec.event_key
        rec['scenario_label']=LABELS[block];rec['n_windows_expected']=k
        rec['n_valid_windows']=nvalid;rec['n_windows_observed']=nvalid
        rec['n_zero_no_operation']=(av==0).sum(axis=1);rec['n_source_blank']=np.isnan(av).sum(axis=1)
        rec['event_risk_available']=nvalid>0;rec['complete_valid_windows']=nvalid==k
        rec['risk_event_mean']=np.divide(np.where(vv,av,0).sum(axis=1),nvalid,out=np.full(len(av),np.nan),where=nvalid>0)
        rec['event_mean_missing_reason']=np.where(nvalid>0,'','no_valid_risk_window')
        rec['candidate_mat_scenario']=g.scenario.iloc[0];rec['candidate_mat_event_id']=eid
        rec['mat_scenario']=None;rec['mat_event_id']=pd.Series(pd.NA,index=rec.index,dtype='Int64')
        rec['mapping_status']='unresolved_physical_link';rec['physical_join_verified']=False
        for wi in range(1,7):
            rec[f'risk_w{wi}']=np.where(vv[:,wi-1],av[:,wi-1],np.nan) if wi<=k else np.nan
            rec[f'risk_w{wi}_raw']=av[:,wi-1] if wi<=k else np.nan
            rec[f'risk_w{wi}_structural_missing']=wi>k
            design.append({'questionnaire_block':block,'questionnaire_event_id':eid,'window_index':wi,'window_is_applicable':wi<=k,'missing_reason_if_inapplicable':'not_defined_structural' if wi>k else ''})
            if wi<=k:
                wr=rec[['participant_id','questionnaire_block','questionnaire_event_id','scenario','event_id','event_key','event_row_id']].copy()
                ar=av[:,wi-1];vr=vv[:,wi-1]
                wr['window_index']=wi;wr['window_row_id']=wr.event_row_id+f'_W{wi}'
                wr['risk_raw']=ar;wr['risk']=np.where(vr,ar,np.nan);wr['risk_valid']=vr
                wr['missing_reason']=np.select([ar==0,np.isnan(ar),~vr],['zero_no_operation','source_blank','out_of_range'],default='')
                wr['window_is_applicable']=True;wr['time_alignment_status']='discrete_index_only'
                windows.append(wr)
        events.append(rec)
    e=pd.concat(events,ignore_index=True).sort_values(['participant_id','scenario','event_id']).reset_index(drop=True)
    w=pd.concat(windows,ignore_index=True).sort_values(['participant_id','scenario','event_id','window_index']).reset_index(drop=True)
    coverage=e.groupby('participant_id').agg(n_assigned_events=('event_key','size'),n_usable_events=('event_risk_available','sum'),n_valid_risk_ratings=('n_valid_windows','sum'),n_zero_no_operation=('n_zero_no_operation','sum'),risk_person_mean=('risk_event_mean','mean'))
    # Each block receives equal weight, even when zero-coded windows differ by block.
    block_means=e.groupby(['participant_id','scenario']).risk_event_mean.mean().unstack()
    coverage['risk_person_block_balanced_mean']=block_means.mean(axis=1,skipna=True)
    coverage['n_blocks_with_risk']=block_means.notna().sum(axis=1)
    p=p.merge(coverage,on='participant_id',validate='one_to_one');p['has_valid_risk']=p.n_valid_risk_ratings.gt(0)
    p['analysis_exclusion_reason']=np.where(p.has_valid_risk,'','no_nonzero_video_risk_ratings')
    e=e.merge(p,on='participant_id',validate='many_to_one',suffixes=('','_participant'))
    w=w.merge(p[['participant_id','country','duration_qc_pass','gender']],on='participant_id',validate='many_to_one')
    for name,df in [('source_participants',source_p),('participants',p),('events',e),('windows',w),('event_assignment',pd.concat(assignments,ignore_index=True)),('window_design',pd.DataFrame(design))]:save(df,data/f'{name}.parquet')
    country=p.groupby('country',dropna=False).agg(n_completed_licensed=('participant_id','size'),n_with_valid_risk=('has_valid_risk','sum'),n_valid_risk_ratings=('n_valid_risk_ratings','sum'),n_zero_no_operation=('n_zero_no_operation','sum'),n_usable_events=('n_usable_events','sum'),duration_qc_pass_n=('duration_qc_pass','sum')).reset_index()
    country['small_n_flag']=country.n_with_valid_risk.lt(30);country['flag_role']='descriptive_flag_only_no_exclusion'
    country.sort_values('n_completed_licensed',ascending=False).to_csv(tables/'d2_country_coverage.csv',index=False)
    flow=pd.DataFrame([('raw_respondents',len(source_p)),('completed_licensed',len(p)),('all_countries',p.country.nunique()),('respondents_with_nonzero_risk',int(p.has_valid_risk.sum())),('assigned_events',len(e)),('events_with_nonzero_risk',int(e.event_risk_available.sum())),('administered_window_slots',len(w)),('zero_no_operation_windows',int(w.risk_raw.eq(0).sum())),('valid_nonzero_risk_ratings',int(w.risk_valid.sum())),('duration_flag_sensitivity_only',int(p.duration_qc_pass.sum()))],columns=['measure','n'])
    flow.to_csv(tables/'d2_sample_flow.csv',index=False)
    e.groupby(['scenario','n_valid_windows']).size().rename('n_events').reset_index().to_csv(tables/'d2_event_window_completeness.csv',index=False)
    w.groupby(['scenario','missing_reason']).size().rename('n_windows').reset_index().to_csv(tables/'d2_risk_missingness.csv',index=False)
    pd.DataFrame(codebook).to_csv(tables/'d2_source_item_codebook.csv',index=False)
    validation={'completed_licensed_participants':len(p),'countries_in_primary_cohort':int(p.country.nunique()),'countries_with_nonzero_risk':int(p.loc[p.has_valid_risk,'country'].nunique()),'participants_with_nonzero_risk':int(p.has_valid_risk.sum()),'assigned_events':len(e),'usable_events':int(e.event_risk_available.sum()),'administered_window_slots':len(w),'risk_zero_no_operation':int(w.risk_raw.eq(0).sum()),'valid_nonzero_ratings':int(w.risk_valid.sum()),'risk_source_blanks':int(w.missing_reason.eq('source_blank').sum()),'duration_qc_not_applied_to_primary':True,'no_country_exclusion':True,'trust_and_acceptance_zero_is_valid':True,'duration_sensitivity_thresholds':[float(qlo),float(qhi)]}
    assert len(p)==2164 and p.participant_id.is_unique and e.event_row_id.is_unique and w.window_row_id.is_unique
    assert len(e)==len(p)*16 and e.groupby(['participant_id','scenario']).size().eq(4).all()
    assert w.loc[w.risk_valid,'risk'].between(1,10).all() and w.loc[w.risk_raw.eq(0),'risk'].isna().all()
    assert len(w)==len(p)*84 and w.risk_valid.sum()+w.risk_raw.eq(0).sum()+w.missing_reason.eq('source_blank').sum()==len(w)
    reread=w.groupby('event_row_id').risk.mean().reindex(e.event_row_id)
    assert np.allclose(reread,e.risk_event_mean,equal_nan=True)
    assert np.array_equal(p.age,2023-p.birth_year) and np.array_equal(p.driving_exp_years,2023-p.license_year)
    trustcols=[f'{family}_{time}_{item}' for time in ['pre','post'] for family,item in [('trust','overall'),('trust','acc'),('trust','lks'),('accept','delegate'),('accept','monitor_raw'),('accept','distract')]]
    assert p[trustcols].notna().all().all() and p[trustcols].min().min()==0 and p[trustcols].max().max()==10
    after={name:digest(path) for name,path in raw_files.items()};assert before==after
    validation['all_assertions_passed']=True;validation['raw_hashes_unchanged']=True
    manifest={'raw_before':before,'raw_after':after,'raw_unchanged':True,'risk_column_mapping_sha256':digest(colmap_path),'builder_sha256':digest(Path(__file__)),'data_sha256':{f.name:digest(f) for f in data.glob('*.parquet')}}
    (reports/'d2_source_manifest.json').write_text(json.dumps(manifest,indent=2))
    (reports/'d2_build_validation.json').write_text(json.dumps(validation,indent=2))
    (reports/'d2_data_schema.json').write_text(json.dumps({f.stem:{c:str(t) for c,t in pd.read_parquet(f).dtypes.items()} for f in data.glob('*.parquet')},indent=2))
    (reports/'d2_data_rules.md').write_text(f'''# Round 2 data rules

User-confirmed interpretation: video risk 0 denotes no operation and is missing for risk analysis; trust/acceptance pre/post 0 is valid. No source value is overwritten. windows.risk_raw retains 0 and windows.risk is null at those slots, with missing_reason=zero_no_operation.

Primary cohort: all {len(p):,} completed/licensed respondents, all {p.country.nunique()} reported countries. Country missingness would be labelled Unknown. Neither country sample-size thresholds nor duration-percentile exclusions are applied. duration_qc_pass retains the previous rule for sensitivity only. small_n_flag identifies country n<30 for interpretation, never exclusion.

Only administered questionnaire events produce event/window rows. event_assignment explicitly identifies unassigned events as structural, and window_design identifies the structurally absent sixth window in B2–B4. A zero is not treated as structural. All-zero administered events remain in events with event_risk_available=False. risk_event_mean averages only valid >0 windows. n_valid_windows (alias n_windows_observed) records its denominator; partial-window events are allowed and must not be interpreted as complete trajectories.

Stable participant_id is the original Excel row. event_row_id and window_row_id uniquely identify observations. No personal external identifiers are copied. scenario stores questionnaire block B1–B4. Author-confirmed labels are LC/SVM/HB/MB; the original candidate physical-join flags are retained for provenance, not silently promoted. Controlled parameter analyses use the separate verified design table. No presentation order is invented. Legacy rating-MAT reconstructions are not imported.

Actual retained nonzero video ratings: {int(w.risk_valid.sum()):,}; countries with at least one valid rating: {p.loc[p.has_valid_risk,'country'].nunique()}. Thus the historical '140K+/17 countries' wording must be reconciled to these exact definitions rather than retained automatically.

Trust items remain three individual 0–10 questions (overall, ACC, LKS), with three acceptance-related items. Monitor raw is need for supervision; monitor_reversed is 10−raw. mean3 is descriptive only. All deltas use post minus pre. Original 2023 age/experience values are retained. Source hashes are verified before and after.
''')
    print(json.dumps(validation,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT);args=ap.parse_args();main(args.root)
