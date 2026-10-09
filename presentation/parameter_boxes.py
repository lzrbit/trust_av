"""Draw four native vector boxplot figures from saved v8 summary tables.

No statistical estimation. Each scenario has three factors. Letters appear
below axes; factor names are specified in captions and units accompany legend levels.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from pypdf import PdfReader


COLORS=['#0072B2','#E69F00','#009E73','#AA6A9F']
FACTORS={'LC':['lateral_behaviour','acc_style','design_distance'], 'SVM':['design_speed','design_braking','design_distance'], 'HB':['design_speed','design_braking','design_distance'], 'MB':['design_speed','design_braking','design_distance']}
FACTOR_NAMES={'lateral_behaviour':'Lateral behaviour','acc_style':'ACC style','design_speed':'Cruising speed (km/h)','design_braking':'Braking intensity (m/s²)','design_distance':'Distance (m)'}
DISTANCE_NAMES={'LC':'Neighbour merging distance (m)','SVM':'Ego merging distance (m)','HB':'Initial braking distance (m)','MB':'Merging distance (m)'}


def draw_factor_panel(ax, boxes, tests, scenario, factor, *, legend=True, xlabel=True, ylabel=True, annotate=True):
    """Draw saved quantiles with exact 1.5-IQR observed whiskers and all fliers.

    Accepts either the full tables or scenario/factor subsets. Returns metadata.
    Geometry stays in the caller's axes, so ROOT can use the same function in h.
    """
    b=boxes[boxes.display_scenario.eq(scenario)&boxes.factor.eq(factor)].copy()
    t=tests[tests.display_scenario.eq(scenario)&tests.factor.eq(factor)].copy()
    levels=b[['level_order','level','level_label']].drop_duplicates().sort_values('level_order'); count=len(levels)
    offsets=(np.arange(count)-(count-1)/2)*(.70/count); width=.58/count
    for r in b.itertuples():
        pos=r.clip+offsets[r.level_order-1]
        stats={'med':r.median,'q1':r.q1,'q3':r.q3,'whislo':r.whisker_low,'whishi':r.whisker_high,'fliers':json.loads(r.fliers_json)}
        color=COLORS[r.level_order-1]
        ax.bxp([stats],positions=[pos],widths=width,patch_artist=True,showfliers=True,manage_ticks=False,boxprops={'facecolor':color,'alpha':.28,'edgecolor':color,'linewidth':.6},medianprops={'color':color,'linewidth':.9},whiskerprops={'color':color,'linewidth':.55},capprops={'color':color,'linewidth':.55},flierprops={'marker':'.','markersize':1.1,'alpha':.35,'markeredgewidth':0,'markerfacecolor':color,'rasterized':False})
    comparisons=t[['reference_order','comparison_order']].drop_duplicates().sort_values(['comparison_order','reference_order'])
    if annotate:
        for ci,r in enumerate(comparisons.itertuples()):
            for row in t[t.reference_order.eq(r.reference_order)&t.comparison_order.eq(r.comparison_order)].itertuples():
                x1=row.clip+offsets[r.reference_order-1]; x2=row.clip+offsets[r.comparison_order-1]; y=10.45+.83*ci
                ax.plot([x1,x1,x2,x2],[y-.10,y,y,y-.10],color='#49545C',linewidth=.4,clip_on=False)
                ax.text((x1+x2)/2,y+.03,row.significance,ha='center',va='bottom',fontsize=6)
    ax.set_xlim(.45,float(b['clip'].max())+.55); ax.set_ylim(.7,10.65+.83*len(comparisons) if annotate else 10.3)
    ax.set_xticks(sorted(b['clip'].unique())); ax.set_yticks([1,3,5,7,9]); ax.set_xlabel('Clip' if xlabel else '',labelpad=2); ax.set_ylabel('Participant mean risk' if ylabel else '',labelpad=2)
    if legend:
        units={'design_speed':' km/h','design_braking':' m/s²','design_distance':' m'}
        labels=[str(r.level_label)+units.get(factor,'') for r in levels.itertuples()]
        handles=[Patch(facecolor=COLORS[i],alpha=.4,edgecolor=COLORS[i],label=label) for i,label in enumerate(labels)]
        ax.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,1.015),ncol=2 if count==4 else count,columnspacing=.7,handlelength=1.2,handletextpad=.35,borderaxespad=0.,labelspacing=.35)
    assert ax.get_title()==''
    return {'scenario':scenario,'factor':factor,'box_count':len(b),'n_participants_min':int(b.n_participants.min()),'n_participants_max':int(b.n_participants.max()),'fliers_displayed':int(b.n_fliers.sum()),'paired_tests':len(t),'family_size':195,'significance_counts':t.significance.value_counts().to_dict()}
