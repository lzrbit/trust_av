"""Replot selected public aggregate tables; not a complete manuscript rebuild.

Individual flier values and other participant-level inputs are intentionally
not distributed. The HB preview therefore hides fliers and states that omission.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from settings import WORKSPACE,REPO

def main():
 out=WORKSPACE/'outputs/public_aggregate_figures';out.mkdir(parents=True,exist_ok=True)
 source=REPO/'data/aggregate'
 boxes=pd.read_csv(source/'parameter_boxstats.csv');boxes=boxes[boxes.display_scenario.eq('HB') & boxes.factor.eq('design_distance')]
 fig,ax=plt.subplots(figsize=(8,3.5));colours=['#3b78a4','#e59846','#ac4655']
 for j,level in enumerate([5,15,25]):
  group=boxes[boxes.level.eq(str(level)) if boxes.level.dtype==object else boxes.level.eq(level)].sort_values('clip')
  stats=[dict(q1=r.q1,med=r.median,q3=r.q3,whislo=r.whisker_low,whishi=r.whisker_high,fliers=[]) for r in group.itertuples()]
  plotted=ax.bxp(stats,positions=group['clip'].to_numpy()+(j-1)*.24,widths=.2,showfliers=False,patch_artist=True,manage_ticks=False)
  for patch in plotted['boxes']:patch.set_facecolor(colours[j]);patch.set_alpha(.55)
  ax.plot([],[],color=colours[j],label=f'{level} m')
 ax.set(xticks=range(1,6),xlabel='Clip index',ylabel='Participant mean risk',ylim=(.5,10.5),title='HB initial braking distance: participant-first boxes\nIndividual outlier points omitted from public preview')
 ax.legend(frameon=False,ncols=3);fig.tight_layout();fig.savefig(out/'hb_boxes.png',dpi=160);plt.close(fig)
 performance=pd.read_csv(source/'risk_prediction.csv');selected=performance[performance.model.eq('LightGBM')]
 fig,ax=plt.subplots(figsize=(8,3.5));x=np.arange(len(selected));y=selected.r2.to_numpy()
 ax.errorbar(x,y,yerr=np.row_stack([y-selected.r2_ci_low,selected.r2_ci_high-y]),fmt='o',capsize=3)
 ax.axhline(0,c='grey',lw=.7);ax.set(xticks=x,xticklabels=selected.protocol,ylabel='Held-out R²')
 ax.tick_params(axis='x',rotation=15);fig.tight_layout();fig.savefig(out/'risk_generalization.png',dpi=160);plt.close(fig)
 if (source/'prepost.csv').exists():
  data=pd.read_csv(source/'prepost.csv');fig,ax=plt.subplots(figsize=(8,3.5));x=np.arange(len(data))
  for label,shift in [('pre',-.12),('post',.12)]:
   y=data[label+'_mean'].to_numpy();ax.errorbar(x+shift,y,yerr=np.row_stack([y-data[label+'_ci_low'],data[label+'_ci_high']-y]),fmt='o',label=label,capsize=2)
  ax.set(xticks=x,xticklabels=data.label,ylabel='Mean score and 95% participant-bootstrap CI');ax.tick_params(axis='x',rotation=18);ax.legend();fig.tight_layout();fig.savefig(out/'prepost.png',dpi=160);plt.close(fig)
 print(f'Saved aggregate previews in {out}')
if __name__=='__main__':main()
