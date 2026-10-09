"""Regenerate sample/accounting summaries from authorized derived tables.

This is the same aggregation used by build_data, without requiring the original
questionnaire workbook. No participant record is added to the public repository.
"""
import json
import pandas as pd
from settings import WORKSPACE,DATA_DIR

def main():
    tab=WORKSPACE/'outputs/core/tables';tab.mkdir(parents=True,exist_ok=True)
    p,e,w,s=[pd.read_parquet(DATA_DIR/(name+'.parquet')) for name in ['participants','events','windows','source_participants']]
    country=p.groupby('country',dropna=False).agg(n_completed_licensed=('participant_id','size'),n_with_valid_risk=('has_valid_risk','sum'),n_valid_risk_ratings=('n_valid_risk_ratings','sum'),n_zero_no_operation=('n_zero_no_operation','sum'),n_usable_events=('n_usable_events','sum'),duration_qc_pass_n=('duration_qc_pass','sum')).reset_index()
    country['small_n_flag']=country.n_with_valid_risk.lt(30);country['flag_role']='descriptive_flag_only_no_exclusion'
    country.sort_values('n_completed_licensed',ascending=False).to_csv(tab/'d2_country_coverage.csv',index=False)
    flow=pd.DataFrame([('raw_respondents',len(s)),('completed_licensed',len(p)),('all_countries',p.country.nunique()),('respondents_with_nonzero_risk',int(p.has_valid_risk.sum())),('assigned_events',len(e)),('events_with_nonzero_risk',int(e.event_risk_available.sum())),('administered_window_slots',len(w)),('zero_no_operation_windows',int(w.risk_raw.eq(0).sum())),('valid_nonzero_risk_ratings',int(w.risk_valid.sum())),('duration_flag_sensitivity_only',int(p.duration_qc_pass.sum()))],columns=['measure','n'])
    flow.to_csv(tab/'d2_sample_flow.csv',index=False)
    e.groupby(['scenario','n_valid_windows']).size().rename('n_events').reset_index().to_csv(tab/'d2_event_window_completeness.csv',index=False)
    w.groupby(['scenario','missing_reason']).size().rename('n_windows').reset_index().to_csv(tab/'d2_risk_missingness.csv',index=False)
    assert len(p)==2164 and p.country.nunique()==29 and w.risk_valid.sum()==179966
    print(json.dumps({'status':'pass','tables':4,'participants':len(p),'countries':p.country.nunique()}))
if __name__=='__main__':main()
