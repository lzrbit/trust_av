"""Small synthetic checks. No participant source data are read or distributed."""
import argparse,importlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from analyse_dynamics import bootstrap_means,profile_matrix

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--with-model',action='store_true');args=ap.parse_args()
 values=np.array([[2.,np.nan],[2.,5.],[2.,5.],[2.,5.]])
 ci,draws=bootstrap_means(values,seed=11,b=64)
 np.testing.assert_allclose(ci,[[2.,5.],[2.,5.]])
 rows=[]
 for person in range(30):
  for block,nclip in [('B1',6),('B2',5),('B3',5),('B4',5)]:
   for clip in range(1,nclip+1):
    for event in [1,2]:rows.append(dict(participant_id=person,scenario=block,window_index=clip,risk=person*.01+clip+event))
 mat=profile_matrix(pd.DataFrame(rows),range(30));assert mat.shape==(30,21)
 assert np.isclose(mat.loc[0,('B1',1)],2.5)
 checks=['participant bootstrap preserves constants with missing observations','21-position profile averages repeated events before between-person summaries']
 if args.with_model:
  m=importlib.import_module('model_shap');records=[]
  for person in range(30):
   for block in ['B1','B2','B3','B4']:
    for event in range(10):records.append(dict(participant_id=person,country=f'Synthetic_{person%3}',scenario=block,event_key=f'{block}_E{event:02d}',row_key=f'{person}_{block}_{event}'))
  data=pd.DataFrame(records);splits,_,_=m.make_splits(data);coverage={}
  for protocol,fold,tr,te in splits:
   m.audit_split(data,protocol,fold,tr,te);coverage.setdefault(protocol,np.zeros(len(data),int))[te]+=1
  assert len(coverage)==4 and all((a==1).all() for a in coverage.values())
  # Test fitting/SHAP on synthetic values only, not the study model.
  x=pd.DataFrame({'age':np.arange(120)%25+20.,'country':['Synthetic_A','Synthetic_B']*60})
  prep=m.make_preprocessor(['age'],['country']);train=prep.fit_transform(x.iloc[:100]);test=prep.transform(x.iloc[100:])
  model=m.lgb.LGBMRegressor(n_estimators=8,num_leaves=5,min_child_samples=5,n_jobs=2,verbosity=-1).fit(train,np.sin(np.arange(100)/10))
  phi=model.predict(test,pred_contrib=True);np.testing.assert_allclose(phi.sum(axis=1),model.predict(test),atol=1e-10)
  checks+=['all four generalization protocols have one test prediction per row and declared disjoint groups','synthetic LightGBM TreeSHAP additivity']
 print(json.dumps({'status':'pass','data':'synthetic only','checks':checks},indent=2))
if __name__=='__main__':main()
