from pathlib import Path
import argparse, csv, hashlib, json, shutil, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap, LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from scipy.cluster.hierarchy import linkage, to_tree, leaves_list
from pypdf import PdfReader
from PIL import Image

from .base import RenderBase

BLUE,TEAL,ORANGE,PURPLE='#0072B2','#009E73','#E69F00','#AA6A9F'
COLORS=[BLUE,ORANGE,TEAL,PURPLE]
GREY='#707A83'; LIGHT='#D9DFE3'; INK='#25333B'
NUMERIC=['age','driving_exp_years','trust_pre_overall','trust_pre_acc','trust_pre_lks','accept_pre_delegate','accept_pre_monitor_raw','accept_pre_distract']
SHORT={'United Kingdom of Great Britain and Northern Ireland':'United Kingdom','United States of America':'United States'}
LABELS={'age':'Age','driving_exp_years':'Driving experience','trust_pre_overall':'Overall trust','trust_pre_acc':'ACC trust','trust_pre_lks':'LKS trust','accept_pre_delegate':'Delegation','accept_pre_monitor_raw':'Supervision need','accept_pre_distract':'Non-driving activities','drive_style':'Driving style','km_12m':'Recent driving distance','know_ad':'Automation knowledge','lks_dist':'LKS use distance','acclks_years':'ACC/LKS use duration','km_life':'Lifetime driving distance','acc_dist':'ACC use distance','education':'Education'}
LABELS.update({'event_key':'Event identity','scenario':'Scenario','country':'Reported country','drive_type':'Usual driving type','gender':'Reported gender','drive_freq_12m':'Driving frequency','acc_years':'ACC use duration','lks_years':'LKS use duration','acclks_dist':'ACC/LKS use distance','own_car':'Car ownership'})
FAMILY={'Questionnaire identity':'Stimulus identity','Reported country':'Reported country','Initial trust':'Initial trust','Initial acceptance items':'Initial acceptance','Automation experience':'Automation experience','Demographics and experience':'Demographics / experience','Driving history and habits':'Driving habits'}
FAMILY_SHORT={'Questionnaire identity':'Stimulus','Reported country':'Country','Initial trust':'Trust','Initial acceptance items':'Acceptance','Automation experience':'Automation use','Demographics and experience':'Demographics','Driving history and habits':'Driving habits'}
FAMILY_COLORS=['#B2BDC5',BLUE,TEAL,ORANGE,PURPLE,'#56B4E9','#D55E00']
PROTOCOLS=['participant_5fold','leave_country_out','event_group_5fold','crossed_participant_event_5x5']
PROTOCOL_LABELS=['New participants','Held-out country','New events','New people + events']

SCENARIO_LABELS={'B1':'LC','B2':'SVM','B3':'HB','B4':'MB'}
def display_label(text):
    text=re.sub(r'\bB[1-4]\b',lambda m:SCENARIO_LABELS[m.group()],str(text))
    for old,new in [('blocks','scenarios'),('block','scenario'),('Blocks','Scenarios'),('Block','Scenario')]:text=re.sub(r'\b'+old+r'\b',new,text)
    return text

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class FigureBuilder(RenderBase):

    def sankey(self,ax,item):
        t=self.table('d2_item_transitions_fixed_bands');t=t[(t.family=='trust')&(t.item==item)]
        bins=['0–3','4–6','7–10'];counts=t.pivot(index='pre_bin',columns='post_bin',values='n').reindex(index=bins,columns=bins).to_numpy()
        total=counts.sum();scale=.78/total;gap=.048;col=[BLUE,ORANGE,TEAL]
        def starts(s):return np.r_[.025,.025+np.cumsum(s[:-1])*scale+np.arange(1,3)*gap]
        left=starts(counts.sum(1));right=starts(counts.sum(0));lc=left.copy();rc=right.copy()
        for a in range(3):
            for b in range(3):
                h=counts[a,b]*scale;v=[(.18,lc[a]),(.42,lc[a]),(.58,rc[b]),(.82,rc[b]),(.82,rc[b]+h),(.58,rc[b]+h),(.42,lc[a]+h),(.18,lc[a]+h),(.18,lc[a])]
                code=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY]
                ax.add_patch(PathPatch(MPath(v,code),facecolor=col[a],edgecolor='none',alpha=.3));lc[a]+=h;rc[b]+=h
        for j in range(3):
            for x,y,n,align,tx in [(.14,left[j],counts[j].sum(),'right',.115),(.82,right[j],counts[:,j].sum(),'left',.885)]:
                ax.add_patch(Rectangle((x,y),.04,n*scale,facecolor=col[j],edgecolor='none'))
                ax.text(tx,y+n*scale/2,f'{bins[j].replace("–","-")}\n{n/total:.0%}',ha=align,va='center',fontsize=6.5)
        ax.text(.16,.99,'Pre',ha='center',fontsize=6.5);ax.text(.84,.99,'Post',ha='center',fontsize=6.5)
        ax.set(xlim=(-.09,1.09),ylim=(-.015,1.065));ax.axis('off')

    def parameters(self):
        f=self.figure(175);source=self.input('controlled_parameter_clip_profiles.csv');d=pd.read_csv(source);d=d[d.analysis_variant=='assigned_design_primary'];assert len(d)==189
        setups=[('B1','LC',[('design_distance','Neighbour merging distance (m)'),('lateral_behaviour','Lateral behaviour'),('acc_style','ACC style')]),('B2','SVM',[('design_distance','Ego merging distance (m)'),('design_speed','Cruising speed (km/h)'),('design_braking','Braking (m/s²)')]),('B3','HB',[('design_distance','Initial braking distance (m)'),('design_speed','Cruising speed (km/h)'),('design_braking','Braking (m/s²)')]),('B4','MB',[('design_distance','Merging distance (m)'),('design_speed','Cruising speed (km/h)'),('design_braking','Braking (m/s²)')])]
        inventories=[];shown=0
        assert d.ci_low.min()>1 and d.ci_high.max()<8
        for row,(internal,display,factors) in enumerate(setups):
            for column,(factor,title) in enumerate(factors):
                letter='abcdefghijkl'[row*3+column];x=14+60*column;y=140-41*row;ax=self.ax(f,x,y,42,22)
                q=d[(d.scenario==internal)&(d.factor==factor)].sort_values(['level_order','clip']);assert not q.empty and q.display_scenario.eq(display).all();shown+=len(q)
                for (order,label),color in zip(q[['level_order','level_label']].drop_duplicates().itertuples(index=False,name=None),COLORS):
                    level=q[q.level_order==order].sort_values('clip');label=str(label)
                    if factor in ['design_distance','design_speed','design_braking']:label=label.split()[0].replace('-','−')
                    ax.fill_between(level['clip'],level.ci_low,level.ci_high,color=color,alpha=.15,lw=0)
                    ax.plot(level['clip'],level['mean'],'o-',color=color,ms=2.3,lw=.85,label=label)
                ax.set(ylim=(1,8),yticks=[1,3,5,7],xticks=range(1,7 if internal=='B1' else 6));ax.tick_params(labelsize=6.5)
                if column==0:ax.set_ylabel('Mean risk')
                if row==3:ax.set_xlabel('Clip position')
                ax.legend(loc='lower center',bbox_to_anchor=(.5,1.03),ncol=2 if factor=='lateral_behaviour' else 3,fontsize=6,handlelength=1.0,handletextpad=.28,columnspacing=.65,borderaxespad=0,labelspacing=.25)
                self.heading(f,letter,f'{display} · {title}',3+60*column,173-41*row)
                self.panel(2,letter,f'{display}: {title}',['saved/controlled_parameter_clip_profiles.csv'],f'Assigned-design primary subset only; display scenario {display}; factor {factor}. Available event ratings are averaged within participant, factor level and clip before equal weighting of contributing people. Groups can share participants and other design factors are marginalised; all reviewed levels are displayed.','Saved pointwise 95% participant-bootstrap mean intervals, 2000 draws. Shaded intervals are not sample SD and not simultaneous intervals.',n=f"varies from {int(q.n_participants.min())} to {int(q.n_participants.max())}")
                inventories.append({'panel':letter,'scenario':internal,'display_scenario':display,'factor':factor,'factor_label':title,'rows':len(q),'n_levels':q.level.nunique(),'n_clips':q['clip'].nunique(),'x_mm':x,'y_mm':y,'width_mm':42,'height_mm':22,'y_min':1,'y_max':8,'analysis_variant':'assigned_design_primary'})
        assert shown==189 and len(inventories)==12
        pd.DataFrame(inventories).to_csv(self.dest/'tables/figure_2_parameter_panel_inventory.csv',index=False)
        self.save(f,'figure_2_parameters')

    def radial(self,ax):
        d=self.table('d2_country_position_profiles');countries=sorted(d.country.unique())
        wide=d.pivot(index='country',columns='position',values='country_centered_mean').reindex(countries)
        z=linkage(wide.to_numpy(),method='ward',optimal_ordering=True);tree=to_tree(z);leaf_order=leaves_list(z).tolist()
        stability=self.table('d2_country_tree_stability');saved={frozenset(r.member_countries.split(' | ')):r for r in stability.itertuples()}
        clades={}
        def memberships(node):
            if node.is_leaf():return frozenset([countries[node.id]])
            s=memberships(node.left)|memberships(node.right);clades[node.id]=s;return s
        memberships(tree);assert set(clades.values())==set(saved), 'Reconstructed country clades differ from saved analysis'
        angles={i:np.pi/2-(j+.5)*2*np.pi/len(countries) for j,i in enumerate(leaf_order)}
        n=d.groupby('country').n_participants_country.first().to_dict()
        def draw(node):
            if node.is_leaf():return angles[node.id],1.
            ta,ra=draw(node.left);tb,rb=draw(node.right);t=(ta+tb)/2;r=.10+.90*(1-node.dist/tree.dist)
            for theta,rr in [(ta,ra),(tb,rb)]:ax.plot([r*np.cos(theta),rr*np.cos(theta)],[r*np.sin(theta),rr*np.sin(theta)],color='#A6B4BC',lw=.65)
            ts=np.linspace(ta,tb,60);ax.plot(r*np.cos(ts),r*np.sin(ts),color='#A6B4BC',lw=.65)
            row=saved[clades[node.id]]
            if node.id!=tree.id and pd.notna(row.bootstrap_clade_recovery) and row.bootstrap_clade_recovery>=.5:
                ax.text(r*np.cos(t),r*np.sin(t),f'{row.bootstrap_clade_recovery:.0%}',ha='center',va='center',fontsize=6.5,bbox={'fc':'white','ec':'none','pad':.3})
            return t,r
        draw(tree)
        for i,country in enumerate(countries):
            t=angles[i];x,y=np.cos(t),np.sin(t);color=TEAL if n[country]>=30 else ('#B95D2B' if n[country]==1 else GREY)
            ax.scatter([x],[y],s=6+2*np.log10(n[country]),color=color,zorder=3)
            deg=np.degrees(t);right=np.cos(t)>=0
            rotation=deg if right else deg+180
            ax.text(1.10*x,1.10*y,f'{SHORT.get(country,country)}  {n[country]}',rotation=rotation,rotation_mode='anchor',ha='left' if right else 'right',va='center',fontsize=6,color=color)
        ax.set(xlim=(-1.85,1.85),ylim=(-1.85,1.85));ax.set_aspect('equal');ax.axis('off')
        return sorted([list(x) for x in clades.values()])

    def dynamics(self):
        f=self.figure(170);d=self.table('a2_position_means')
        dispersion=pd.read_csv(self.input('participant_position_dispersion.csv'))
        check=d.merge(dispersion,on=['scenario','position'],suffixes=('_original','_display'),validate='one_to_one');assert len(check)==21
        for col in ['mean','ci_low','ci_high']:assert np.allclose(check[col+'_original'],check[col+'_display'],rtol=0,atol=1e-12)
        assert check.n_participants.eq(check.n).all()
        d=d.merge(dispersion[['scenario','position','sd']],on=['scenario','position'],validate='one_to_one')
        self.reportdir.joinpath('figure_2_sd_validation.json').write_text(json.dumps({'positions':21,'mean_n_CI_equal_to_reviewed_source':True,'v5_estimation_performed':False,'SD_reused_from_v4':True,'scenario_display_mapping':SCENARIO_LABELS},indent=2)+'\n')
        for j,(letter,block,col) in enumerate(zip('abcd',['B1','B2','B3','B4'],COLORS)):
            x=13+44*j;ax=self.ax(f,x,140,30,20);s=d[d.scenario==block]
            ax.fill_between(s.position,s['mean']-s.sd,s['mean']+s.sd,color=col,alpha=.16,lw=0);ax.errorbar(s.position,s['mean'],yerr=[s['mean']-s.ci_low,s.ci_high-s['mean']],fmt='o-',ms=2.8,capsize=1.5,color=col,lw=.9)
            ax.set(ylim=(0,10),yticks=[0,5,10],xticks=s.position,xlabel='Clip position')
            if j==0:ax.set_ylabel('Mean risk')
            self.heading(f,letter,block,x-10,168)
            self.panel(3,letter,f'{block}: risk by clip position',['tables/a2_position_means.csv','saved/participant_position_dispersion.csv'],'Participant means across assigned valid event ratings at each clip position, then participant average. Pale band = mean ± one sample SD of participant x_ibk values (ddof=1), not SE.','Deep-colour error bars reuse pointwise 95% participant bootstrap CIs for the mean, 2000 draws. SD band represents between-person dispersion, not uncertainty of the mean.',n='varies by clip; saved source gives n')
        self.text(f,3,128,'Shading: ±1 participant SD; bars: 95% CI of the mean',fontsize=6.5)
        countries=self.table('d2_country_coverage').country.tolist();c=self.table('a2_country_profiles').set_index('country').reindex(countries)
        columns=[x for x in c.columns if x.startswith('B')];matrix=c[columns].to_numpy(copy=True)
        for block in ['B1','B2','B3','B4']:
            idx=[i for i,x in enumerate(columns) if x.startswith(block)];matrix[:,idx]-=matrix[:,idx].mean(1,keepdims=True)
        vmax=max(abs(matrix).max(),1);ax=self.ax(f,28,22,29,90)
        mesh=ax.pcolormesh(np.arange(22)-.5,np.arange(30)-.5,matrix,cmap='RdBu_r',vmin=-vmax,vmax=vmax,rasterized=False)
        ax.set_ylim(28.5,-.5);ax.set_yticks(range(29),[f'{SHORT.get(x,x)}  {int(c.loc[x,"n_participants"])}' for x in countries]);ax.set_xticks([]);ax.tick_params(length=0,pad=2,labelsize=6)
        for cut in [5.5,10.5,15.5]:ax.axvline(cut,color='white',lw=.8)
        for x,label in [(2.5,'B1'),(8,'B2'),(13,'B3'),(18,'B4')]:ax.text(x,-.08,label,transform=ax.get_xaxis_transform(),ha='center',fontsize=6.5)
        self.cb(f,mesh,(29,8,27,1.8),'Scenario-centred risk',orientation='horizontal',ticks=[-3,0,3]);self.heading(f,'e','Centred profiles',3,122)
        self.panel(3,'e','Country-block-centred profiles',['tables/a2_country_profiles.csv','tables/d2_country_coverage.csv'],'29 countries; identical country order in panels e-g. Each block mean removed, without scaling.',transform='Saved country-position means centred separately within each questionnaire block.')
        for letter,table,x,column,title,xlabel,color,xlim in [('f','a2_country_stimulus_standardized',63,'stimulus_adjusted_difference','Level','Difference',BLUE,(-2.6,2.6)),('g','a2_country_shape_similarity',80,'shape_r','Shape','Correlation',TEAL,(.4,1.04))]:
            s=self.table(table).set_index('country').reindex(countries);ax=self.ax(f,x,22,11,90);y=np.arange(29)
            valid=s.ci_low.notna()&s.ci_high.notna();ax.hlines(y[valid],s.loc[valid,'ci_low'],s.loc[valid,'ci_high'],color=color,lw=.7)
            ax.scatter(s[column],y,s=10,color=[color if c.loc[z,'n_participants']>1 else '#B95D2B' for z in countries],zorder=3)
            ax.set(ylim=(28.5,-.5),xlim=xlim,yticks=[],xlabel=('Risk level' if letter=='f' else 'Shape r'));ax.set_xticks([-2,0,2] if letter=='f' else [.5,1]);ax.tick_params(labelsize=6);ax.spines['left'].set_visible(False)
            if letter=='f':ax.axvline(0,color=LIGHT,lw=.6,zorder=0)
            self.heading(f,letter,title,x-2,122)
            self.panel(3,letter,title,['tables/'+table+'.csv','tables/d2_country_coverage.csv'],'All 29 countries; identical order to panel e. Singleton points retained without inferential intervals.', '2000 participant bootstrap draws; matched-stimulus external reference fixed; singleton intervals not estimable.' if letter=='f' else '1000 within-country participant draws; external template fixed; undefined draws omitted (Denmark 994/1000 valid).')
        self.heading(f,'h','Exploratory Ward tree',97,122)
        self.radial(self.ax(f,96,17,83,105));self.panel(3,'h','Country-centred profile tree',['tables/d2_country_position_profiles.csv','tables/d2_country_tree_stability.csv'],'All 21 clip positions centred together per country; 29 countries retained. Saved bootstrap clade recovery labels shown only where every leaf has n > 1 and recovery >= 50%. Rust singleton labels.', 'Saved 500 accepted within-country participant bootstrap draws; 3 incomplete-profile draws rejected.',transform='Deterministic Ward linkage of saved centred profiles; every reconstructed clade verified equal to saved clade table; circumferential leaf labels are a display layout only.')
        self.save(f,'figure_3_dynamics')

    def shap(self):
        f=self.figure(170);imp=self.table('m2_shap_global_importance');personal=imp[~imp.feature.isin(['scenario','event_key','country'])].head(14).iloc[::-1]
        ax=self.ax(f,42,122,41,38);ax.hlines(range(len(personal)),0,personal.mean_abs_shap,color=BLUE,lw=.7,alpha=.55);ax.scatter(personal.mean_abs_shap,range(len(personal)),s=12,color=BLUE,zorder=3);ax.set_yticks(range(len(personal)),[LABELS.get(x,x) for x in personal.feature]);ax.set_xlabel('Mean |SHAP| (risk points)');ax.xaxis.set_major_locator(MaxNLocator(3));self.heading(f,'a','Personal-attribute attribution',3,168)
        self.panel(4,'a','Personal-attribute attribution',['tables/m2_shap_global_importance.csv'],'Top 14 saved personal-feature mean absolute SHAP values; stimulus and country excluded from panel. 34566 OOF events.')
        sv=pd.read_parquet(self.source/'prediction/m2_oof_shap.parquet');sample=sv.sample(min(6000,len(sv)),random_state=20261001);order=sorted(NUMERIC,key=lambda x:sample[f'shap__{x}'].abs().mean());ax=self.ax(f,126,122,40,38)
        cmap=LinearSegmentedColormap.from_list('shap',[BLUE,'#CC79A7']);rng=np.random.default_rng(20261002)
        for yi,feature in enumerate(order):
            x=sample[f'shap__{feature}'].to_numpy();values=sample[f'value__{feature}'].to_numpy(float);bins=np.round((x-x.min())/max(np.ptp(x),1e-12)*100).astype(int);offsets=np.zeros(len(x))
            for b in np.unique(bins):
                ids=np.flatnonzero(bins==b);rng.shuffle(ids);seq=np.arange(len(ids));offsets[ids]=np.where(seq%2==0,seq/2,-(seq+1)/2)
            offsets*=.32/max(abs(offsets).max(),1);lo,hi=np.nanpercentile(values,[5,95]);v=np.clip((values-lo)/max(hi-lo,1e-12),0,1)
            ax.scatter(x,yi+offsets,c=v,cmap=cmap,vmin=0,vmax=1,s=.8,alpha=.7,lw=0,rasterized=False)
        ax.axvline(0,color=GREY,lw=.5);ax.set_yticks(range(8),[LABELS[x] for x in order]);ax.set(ylim=(-.6,7.6),xlabel='SHAP contribution');ax.xaxis.set_major_locator(MaxNLocator(3))
        cb=self.cb(f,ScalarMappable(norm=Normalize(0,1),cmap=cmap),(170,122,2,38),'',ticks=[0,1]);cb.ax.set_yticklabels(['Low','High']);self.heading(f,'b','Held-out SHAP distribution',98,168)
        self.panel(4,'b','Held-out SHAP distribution',['prediction/m2_oof_shap.parquet'],'Fixed original 6000-row OOF display sample; eight numeric attributes.',transform='Deterministic vertical collision spacing only; feature colours clipped to display 5th/95th percentiles.',n=int(sample.participant_id.nunique()))
        density_audit=[]
        for letter,feature,x,y in [('c','age',14,70),('d','driving_exp_years',58,70),('e','trust_pre_acc',102,70),('f','trust_pre_lks',146,70)]:
            ax=self.ax(f,x,y,24,27)
            if feature in ['age','driving_exp_years']:
                xx=sample[f'value__{feature}'].to_numpy(float);yy=sample[f'shap__{feature}'].to_numpy(float);assert np.isfinite(xx).all() and np.isfinite(yy).all()
                xmin,xmax=xx.min(),xx.max();ymin,ymax=yy.min(),yy.max()
                hx=ax.hexbin(xx,yy,gridsize=22,mincnt=1,extent=(xmin,xmax,ymin,ymax),cmap='Blues',norm=LogNorm(),linewidths=0,rasterized=False)
                assert int(hx.get_array().sum())==6000
                density_audit.append({'panel':letter,'feature':feature,'event_rows':len(xx),'participants':int(sample.participant_id.nunique()),'nonempty_hexagons':len(hx.get_array()),'sum_hex_counts':int(hx.get_array().sum()),'maximum_hex_count':int(hx.get_array().max()),'gridsize':22,'mincnt':1,'colour_normalisation':'LogNorm, independent range for each panel','count_clipping':False,'hex_extent':[xmin,xmax,ymin,ymax],'axis_margins_each_side':0.05,'display_sample_random_state':20261001})
                ax.set(xlim=(xmin-.05*(xmax-xmin),xmax+.05*(xmax-xmin)),ylim=(ymin-.05*(ymax-ymin),ymax+.05*(ymax-ymin)))
                cb=self.cb(f,hx,(x+26,70,1.6,27),'Rows / hex',ticks=[1,10,100]);cb.ax.set_yticklabels(['1','10','100']);cb.ax.minorticks_off()
                pd.DataFrame({feature+'_hex_center':hx.get_offsets()[:,0],'shap_hex_center':hx.get_offsets()[:,1],'event_rows':hx.get_array().astype(int)}).to_csv(self.dest/'tables'/('shap_'+('age' if feature=='age' else 'experience')+'_hexbin_counts.csv'),index=False)
            else:ax.scatter(sample[f'value__{feature}'],sample[f'shap__{feature}'],s=2.3,alpha=.13,color=BLUE,lw=0,rasterized=False)
            ax.axhline(0,color=GREY,lw=.5)
            ax.set_xlabel({'age':'Age (years)','driving_exp_years':'Experience (years)','trust_pre_acc':'ACC trust (pre)','trust_pre_lks':'LKS trust (pre)'}[feature]);ax.set_ylabel('SHAP' if letter=='c' else '');ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(3));self.heading(f,letter,{'age':'Age','driving_exp_years':'Experience','trust_pre_acc':'ACC trust','trust_pre_lks':'LKS trust'}[feature],x-11,y+37)
            self.panel(4,letter,LABELS[feature]+' dependence',['prediction/m2_oof_shap.parquet']+(['saved/shap_'+('age' if feature=='age' else 'experience')+'_hexbin_counts.csv'] if feature in ['age','driving_exp_years'] else []),'Same original fixed 6000-event OOF display sample, 2057 people; saved feature values and SHAP contributions; not intervention effects.',transform='Age/experience: two-dimensional hexbin count, gridsize 22, no smoothing; count total 6000; colour is event rows per hexagon on a logarithmic colour scale. Original scatter data extent plus 5% margins retained.' if feature in ['age','driving_exp_years'] else 'Original individual event rows.',n=int(sample.participant_id.nunique()))
        (self.reportdir/'shap_density_validation.json').write_text(json.dumps({'status':'pass','panels':density_audit,'source_sha256':digest(self.source/'prediction/m2_oof_shap.parquet'),'no_refit_or_new_sample_exclusion':True},indent=2)+'\n')
        pairs=self.table('m2_global_pair_interactions');matrix=np.zeros((8,8))
        for r in pairs.itertuples():
            if r.feature_a in NUMERIC and r.feature_b in NUMERIC:i,j=NUMERIC.index(r.feature_a),NUMERIC.index(r.feature_b);matrix[i,j]=matrix[j,i]=r.mean_abs_pair_contribution
        ax=self.ax(f,31,17,54,29);mesh=ax.pcolormesh(np.arange(9)-.5,np.arange(9)-.5,matrix,cmap='Purples',rasterized=False)
        short=['Age','Exp.','Overall','ACC','LKS','Delegate','Monitor','Non-drive'];ax.set_ylim(7.5,-.5);ax.set_yticks(range(8),short);ax.set_xticks(range(8),short,rotation=45,ha='right');ax.tick_params(length=0,pad=2)
        self.cb(f,mesh,(89,17,2,29),'Mean |pair SHAP|');self.heading(f,'g','True pair contributions',3,55)
        self.panel(4,'g','True pair contributions',['tables/m2_global_pair_interactions.csv'],'400 held-out events from 100 participants; pair = twice off-diagonal TreeSHAP interaction; diagonal zero is display only.',transform='Saved numeric-pair magnitudes in vector cells.',n=100)
        d=pd.read_parquet(self.source/'prediction/m2_sampled_true_interactions.parquet');d=d[(d.feature_a=='age')&(d.feature_b=='driving_exp_years')].groupby('participant_id',as_index=False).agg(value_a=('value_a','first'),value_b=('value_b','first'),pair_contribution=('pair_contribution','mean'))
        lim=max(d.pair_contribution.abs().quantile(.99),1e-5);ax=self.ax(f,126,17,39,29);sc=ax.scatter(d.value_a,d.value_b,c=d.pair_contribution,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),s=7,alpha=.85,lw=0)
        ax.set(xlabel='Age (years)',ylabel='Experience (years)');self.cb(f,sc,(170,17,2,29),'Pair SHAP');self.heading(f,'h','Age × experience',103,55)
        self.panel(4,'h','Age by experience pair contribution',['prediction/m2_sampled_true_interactions.parquet'],'100 held-out participants; saved true pair contributions averaged over sampled events per person.',transform='Original per-participant mean of saved pair SHAP; colour limits at original 99th absolute percentile.',n=100)
        self.save(f,'figure_4_shap')

    def profiles(self):
        f=self.figure(170);s=self.table('m2_profile_attributions');pos=np.arange(len(s));ax=self.ax(f,28,121,55,37)
        for quantity,marker,shift,col,label in [('prediction','o',-.09,BLUE,'Predicted'),('observed','s',.09,INK,'Observed')]:
            ax.vlines(pos+shift,s[f'{quantity}_ci_low'],s[f'{quantity}_ci_high'],color=col,lw=.7);ax.scatter(pos+shift,s[f'mean_{quantity}'],marker=marker,s=13,color=col,facecolors='white' if quantity=='observed' else col,label=label,zorder=3)
        ax.set_xticks(pos,s.profile);ax.set_xlim(-.6,7.6);ax.set_ylabel('Mean operated risk');ax.legend(loc='upper left',handletextpad=.3)
        self.heading(f,'a','Profiles by predicted risk',3,168);self.panel(5,'a','Profile means',['tables/m2_profile_attributions.csv'],'Eight saved condition combinations; sorted by saved prediction mean.','1000 participant-cluster draws of fixed OOF predictions/observations.')
        families=[x.replace('signed__','') for x in s if x.startswith('signed__')];signed=s[[f'signed__{x}' for x in families]].to_numpy().T;lim=max(abs(signed).max(),1e-5)
        ax=self.ax(f,131,121,35,37);mesh=ax.pcolormesh(np.arange(9)-.5,np.arange(8)-.5,signed,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),rasterized=False)
        ax.set_ylim(6.5,-.5);ax.set_yticks(range(7),[FAMILY_SHORT[x] for x in families]);ax.set_xticks(pos,s.profile);ax.tick_params(length=0)
        self.cb(f,mesh,(170,121,2,37),'Signed SHAP');self.heading(f,'b','Signed contributions',98,168)
        self.panel(5,'b','Signed family contributions',['tables/m2_profile_attributions.csv'],'Saved mean signed family SHAP sums; direction preserved.')
        ax=self.ax(f,28,98,55,11);conditions=['ACC trust ≤ 5','LKS trust ≤ 5','Driving experience ≤ 10 years']
        for y,key in enumerate(conditions):
            active=s[key].astype(str).str.lower().eq('true').to_numpy();ax.scatter(pos,np.full(len(pos),y),s=21,c=np.where(active,INK,LIGHT),zorder=3)
        for j,r in s.iterrows():
            active=[i for i,key in enumerate(conditions) if str(r[key]).lower()=='true']
            if len(active)>1:ax.plot([j,j],[min(active),max(active)],color=INK,lw=.8)
        ax.set_yticks(range(3),['ACC ≤ 5','LKS ≤ 5','Experience ≤ 10 y']);ax.set_xticks(pos,[f'{r.profile}\n{r.n_participants}' for r in s.itertuples()]);ax.set(xlim=(-.6,7.6),ylim=(2.7,-.7));ax.tick_params(length=0)
        for spine in ax.spines.values():spine.set_visible(False)
        self.heading(f,'c','Conditions and participant counts',3,115);self.panel(5,'c','Conditions and counts',['tables/m2_profile_attributions.csv'],'Black = stated condition; grey = complement. ACC and LKS trust <=5; experience <=10 years; descriptive thresholds.')
        m=self.table('m2_shap_family_importance').sort_values('mean_abs_grouped_shap');ax=self.ax(f,131,94,42,18)
        ax.barh(range(7),m.mean_abs_grouped_shap,color=BLUE,height=.72);ax.set_yticks(range(7),[FAMILY_SHORT[x] for x in m.family]);ax.set(xlim=(0,.65),xticks=[0,.3,.6],xlabel='')
        self.heading(f,'d','Mean |family SHAP|',98,115);self.panel(5,'d','Global family magnitude',['tables/m2_shap_family_importance.csv'],'Mean absolute signed family sum across all held-out event rows.')
        self.whatif_main_panels(f)
        self.save(f,'figure_5_profiles')

    def trust(self):
        f=self.figure(170);items=['overall','acc','lks'];names=['Overall','ACC','LKS']
        changes=self.table('a2_trust_changes').set_index('item').reindex(items)
        p=pd.read_parquet(self.data/'participants.parquet');frequency=[]
        for item in items:
            delta=p['trust_post_'+item]-p['trust_pre_'+item]
            assert len(delta)==2164 and delta.notna().all() and delta.eq(np.round(delta)).all() and delta.between(-10,10).all()
            counts=delta.value_counts().reindex(range(-10,11),fill_value=0)
            assert int(counts.sum())==2164
            for score,n in counts.items():frequency.append({'item':item,'post_minus_pre':int(score),'n':int(n),'n_participants':2164,'proportion':n/2164})
            assert np.isclose(delta.mean(),changes.loc[item,'delta'],atol=1e-12)
        frequency=pd.DataFrame(frequency);self.dest.joinpath('tables').mkdir(exist_ok=True)
        frequency.to_csv(self.dest/'tables/trust_delta_frequencies.csv',index=False)
        self.reportdir.joinpath('trust_frequency_validation.json').write_text(json.dumps({'n_per_item':frequency.groupby('item').n.sum().to_dict(),'participant_rows':len(p),'mean_matches_saved_changes':True,'new_uncertainty_estimates':False,'bins':'integer post-minus-pre scores -10 through 10; no smoothing'},indent=2)+'\n')
        ax=self.ax(f,17,129,45,30)
        for i,(item,col) in enumerate(zip(items,COLORS)):
            vals=frequency[frequency.item==item].proportion.to_numpy();base=2-i
            ax.stairs(base+vals*1.8,np.arange(-10.5,11.5),baseline=base,fill=True,color=col,alpha=.28,lw=.7)
            ax.stairs(base+vals*1.8,np.arange(-10.5,11.5),baseline=None,color=col,lw=.75)
        ax.axvline(0,color=LIGHT,lw=.5);ax.set(yticks=[2,1,0],yticklabels=names,xlim=(-10.5,10.5),xticks=[-10,0,10],ylim=(-.1,2.8),xlabel='Post − pre score')
        ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False)
        ax.plot([-9,-9],[.10,.46],color=GREY,lw=.6);ax.text(-8.4,.28,'20%',fontsize=6,va='center',color=GREY)
        meanax=self.ax(f,73,129,15,30)
        for i,(item,col) in enumerate(zip(items,COLORS)):
            row=changes.loc[item];meanax.errorbar(row.delta,2-i,xerr=[[row.delta-row.ci_low],[row.ci_high-row.delta]],fmt='o',color=col,ms=2.8,lw=.85,capsize=2)
        meanax.axvline(0,color=LIGHT,lw=.6);meanax.set(xlim=(-.5,.55),ylim=(-.1,2.8),yticks=[],xticks=[-.5,0,.5],xlabel='Mean change');meanax.tick_params(axis='x',labelsize=6)
        meanax.spines['left'].set_visible(False);meanax.set_title('95% CI',fontsize=6.5,pad=3)
        self.heading(f,'a','Trust-rating changes',3,168)
        self.panel(6,'a','Trust-rating changes',['data/participants.parquet','tables/a2_trust_changes.csv','saved/trust_delta_frequencies.csv'],'Three integer post-minus-pre distributions, each n=2164. Original 0-10 item zeros remain valid. Separate aligned mean-change interval inset.','Inset reuses saved pointwise 95% participant bootstrap intervals, 2000 draws. No new uncertainty estimates.',transform='Exact integer frequency counts; step ridgelines use a shared frequency-height factor of 1.8, with a 20% scale bar; no KDE or other smoothing.')
        associations=self.table('a2_trust_risk_associations');features=['mean_risk','excursion','late_change','positive_fraction'];labels=['Mean risk','Excursion','Late change','Response fraction']
        matrix=associations.pivot(index='feature',columns='item',values='estimate').reindex(index=features,columns=items).to_numpy()
        ax=self.ax(f,116,129,49,30);lim=.4
        mesh=ax.pcolormesh(np.arange(4)-.5,np.arange(5)-.5,matrix,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),rasterized=False)
        for i in range(4):
            for j in range(3):ax.text(j,i,f'{matrix[i,j]:+.3f}',ha='center',va='center',fontsize=6.5,color='white' if abs(matrix[i,j])>.25 else INK)
        ax.set(ylim=(3.5,-.5));ax.set_xticks(range(3),names);ax.set_yticks(range(4),labels);ax.tick_params(length=0)
        self.cb(f,mesh,(172,129,2,30),'β',ticks=[-.4,0,.4]);self.heading(f,'b','Adjusted associations',96,168)
        self.panel(6,'b','Adjusted post-trust associations',['tables/a2_trust_risk_associations.csv'],'Twelve saved point estimates per predictor SD, printed to three decimals. Outcome-specific pre-trust, age, experience, gender, country and four risk summaries adjusted. No significance stars.','Pointwise HC1 robust 95% intervals are retained in the source and SI Table S6; heatmap encodes point estimates only.',n=2158)
        source=self.input('within_person_trust_contrast_risk3.csv');s=pd.read_csv(source).set_index('term').reindex(['mean_risk_z','excursion_z','late_change_z']);assert len(s)==3 and s.n.eq(2158).all();ax=self.ax(f,29,78,49,27)
        ax.errorbar(s.estimate,range(3),xerr=[s.estimate-s.ci_low,s.ci_high-s.estimate],fmt='o',ms=3,color=TEAL,lw=.8,capsize=1.5);ax.axvline(0,color=GREY,lw=.5)
        ax.set_yticks(range(3),['Mean risk','Risk excursion','Late change']);ax.set(ylim=(2.5,-.5),xlabel='Trust-change difference per SD',xlim=(-.55,.2),xticks=[-.5,-.25,0]);ax.xaxis.label.set_fontsize(6.5)
        self.heading(f,'c','Within-person trust contrast',3,114)
        self.panel(6,'c','Risk associations with within-person LKS-minus-ACC trust change',['saved/within_person_trust_contrast_risk3.csv'],'Post-review exploratory single direct outcome D = (post LKS − pre LKS) − (post ACC − pre ACC). Three coefficients per predictor SD, jointly adjusted for risk summaries, response fraction, all six original pre-task attitudes, demographics and country. No outcome coefficient is inferred from separate significance tests.','Saved HC1 robust pointwise 95% intervals; BH q across the fixed three-risk-coefficient family of this post-review exploratory model retained in the source table.',n=2158)
        inc=self.table('a2_trust_cv_increment');protocols=[('participant_5fold','New people',TEAL),('leave_country_out','Held-out country',PURPLE)]
        ax=self.ax(f,114,78,43,27)
        for k,(protocol,label,col) in enumerate(protocols):
            s=inc[(inc.protocol==protocol)&(inc.reference_feature_set=='pre_task_plus_response')&(inc.feature_set=='pre_task_plus_both')].set_index('item').reindex(items)
            y=np.arange(3)+(k-.5)*.26
            for j,(_,r) in enumerate(s.iterrows()):
                ax.plot([r.r2_reference,r.r2_expanded],[y[j],y[j]],color=col,lw=.9)
                ax.scatter(r.r2_reference,y[j],s=13,facecolors='white',edgecolors=col,lw=.7,zorder=3)
                ax.scatter(r.r2_expanded,y[j],s=13,color=col,lw=0,zorder=3)
                ax.text(1.05,y[j],f'+{r.delta_r2:.3f}',transform=ax.get_yaxis_transform(),va='center',fontsize=6,color=col)
        ax.set(ylim=(2.6,-.6),xlim=(.255,.372),xticks=[.26,.31,.36],yticks=range(3),yticklabels=names,xlabel='Out-of-fold R²');ax.text(1.05,1.03,'ΔR²',transform=ax.transAxes,fontsize=6.5)
        self.heading(f,'d','Added risk summaries',97,114)
        self.panel(6,'d','Added risk summaries',['tables/a2_trust_cv_increment.csv'],'Six paired R² comparisons: pre-task attributes plus response fraction versus the same baseline plus all risk summaries. Open circle = reference; filled = expanded. Connecting lines link fixed estimates and are not confidence intervals. Paired differences printed to three decimals.','Paired-participant pointwise 95% ΔR² intervals, 2000 fixed-OOF draws, remain in source and SI Table S5; no interval is assigned to the connecting line.')
        mult=self.table('a2_shape_increment_multiplicity_sensitivity');ax=self.ax(f,24,20,40,33)
        for k,(protocol,label,col) in enumerate(protocols):
            s=inc[(inc.protocol==protocol)&(inc.reference_feature_set=='pre_task_plus_mean_risk_response')&(inc.feature_set=='pre_task_plus_both')].set_index('item').reindex(items)
            wide=mult[mult.protocol==protocol].set_index('item').reindex(items);assert np.allclose(s.delta_r2,wide.delta_r2,atol=1e-12)
            y=np.arange(3)+(k-.5)*.27
            ax.hlines(y,wide.ci_low_bonferroni_6,wide.ci_high_bonferroni_6,color=col,lw=.65)
            ax.hlines(y,s.delta_ci_low,s.delta_ci_high,color=col,lw=2.5)
            ax.scatter(s.delta_r2,y,s=9,facecolors='white',edgecolors=col,lw=.7,zorder=3)
        ax.axvline(0,color=GREY,lw=.5);ax.set(ylim=(2.6,-.6),yticks=range(3),yticklabels=names,xlim=(-.006,.0145),xticks=[0,.01],xlabel='Added OOF R²')
        self.heading(f,'e','Shape above mean',3,64);self.text(f,7,57,'Thick: 95%; thin: six-test',fontsize=6)
        self.panel(6,'e','Shape above mean',['tables/a2_trust_cv_increment.csv','tables/a2_shape_increment_multiplicity_sensitivity.csv'],'All six outcome/protocol contrasts shown; pre-task-plus-mean-risk-and-response baseline versus plus both dynamics. Training-only peak selection.','Thick: pointwise 95% paired participant intervals, 2000 fixed-OOF draws. Thin: saved exploratory Bonferroni-six percentile sensitivity, 6000 draws. LOCO LKS sensitivity interval [-0.0001480743,0.0127383183] includes zero.',transform='Nested intervals are two saved uncertainty summaries; no resampling or interval reconstruction.')
        perf=self.table('m2_performance');ax=self.ax(f,83,20,40,33);group=np.arange(4)
        models=[('Training event/block mean','Event/scenario mean',GREY),('Ridge','Ridge',BLUE),('LightGBM','LightGBM',ORANGE)]
        for j,(model,label,col) in enumerate(models):
            s=perf[perf.model==model].set_index('protocol').reindex(PROTOCOLS);x=group+(j-1)*.23
            ax.bar(x,s.r2,width=.20,color=col,label=label,zorder=2)
            ax.errorbar(x,s.r2,yerr=[s.r2-s.r2_ci_low,s.r2_ci_high-s.r2],fmt='none',color=INK,lw=.55,capsize=.9,zorder=3)
        ax.axhline(0,color=GREY,lw=.5);ax.set(xticks=group,xticklabels=['People','Country','Events','People +\nevents'],ylabel='Out-of-fold R²',ylim=(-.015,.275),yticks=[0,.1,.2]);ax.tick_params(axis='x',labelsize=6)
        self.heading(f,'f','Generalisation',74,64)
        self.panel(6,'f','Generalisation targets',['tables/m2_performance.csv'],'Three saved full-model/reference R² results in four protocols; 34566 valid event means per model and protocol. Bars encode fixed R², not participant counts.','Pointwise 95% participant-cluster bootstrap, 1000 fixed-OOF draws.')
        s=self.table('m2_person_increment');ax=self.ax(f,148,20,27,33)
        for j,(model,col) in enumerate([('Ridge',BLUE),('LightGBM',ORANGE)]):
            q=s[s.model==model].set_index('protocol').reindex(PROTOCOLS);y=np.arange(4)+(j-.5)*.28
            ax.barh(y,q.increment_r2,height=.24,color=col)
            ax.errorbar(q.increment_r2,y,xerr=[q.increment_r2-q.person_increment_r2_ci_low,q.person_increment_r2_ci_high-q.increment_r2],fmt='none',color=INK,lw=.55,capsize=1)
        ax.axvline(0,color=GREY,lw=.5);ax.set(ylim=(3.6,-.6),yticks=range(4),yticklabels=['People','Country','Events','People +\nevents'],xlabel='Added R²',xlim=(-.025,.15),xticks=[0,.1]);ax.tick_params(axis='y',labelsize=6)
        self.heading(f,'g','Attribute increments',130,64)
        self.panel(6,'g','Matched personal-attribute increments',['tables/m2_person_increment.csv'],'Full-attribute versus identity-only, same algorithm and splits; eight saved comparisons. Diverging bars originate at zero.','Pointwise 95% paired participant-cluster bootstrap, 1000 fixed-OOF draws.')
        from matplotlib.lines import Line2D
        f.legend([Line2D([0],[0],color=col,lw=2) for _,_,col in protocols],[x[1] for x in protocols],loc='lower left',bbox_to_anchor=(.03,.003),ncol=2,handlelength=1.1,columnspacing=.8,fontsize=6)
        f.legend([Rectangle((0,0),1,1,fc=col) for _,_,col in models],[x[1] for x in models],loc='lower right',bbox_to_anchor=(.99,.003),ncol=3,handlelength=.9,columnspacing=.7,fontsize=6)
        self.save(f,'figure_6_trust')

    def demographic_single(self,field):
        is_age=field=='age';number='S_age' if is_age else 'S_experience';name='figure_s_age' if is_age else 'figure_s_experience'
        f=self.figure(80,86);p=pd.read_parquet(self.data/'participants.parquet');dem=self.table('d2_demographics').set_index('variable')
        bins=np.arange(17.5,75,3) if is_age else np.arange(-.5,58,3);label='Age (years)' if is_age else 'Driving experience (years)'
        ax=self.ax(f,17,17,65,54);ax.hist(p[field].dropna(),bins=bins,color=TEAL,edgecolor='white',linewidth=.5)
        ax.axvline(dem.loc[field,'median'],color=INK,lw=.8,ls='--');ax.set(xlabel=label,ylabel='Participants')
        ax.text(.98,.97,f"n = {int(dem.loc[field,'n']):,}\nMedian = {dem.loc[field,'median']:.0f}",transform=ax.transAxes,ha='right',va='top',fontsize=6.5)
        self.panel(number,'',label+' distribution',['data/participants.parquet','tables/d2_demographics.csv'],'Original completed/licensed participant sample; no age/experience exclusions. Saved median shown.',transform='Original histogram edges reused; one participant per observation. Standalone chart has no panel letter.')
        self.save(f,name)

    def coverage_single(self):
        f=self.figure(85);d=self.table('a2_position_means');ax=self.ax(f,17,19,157,56);positions=np.arange(len(d))
        ax.bar(positions,d.n_participants,color=[COLORS[int(x[-1])-1] for x in d.scenario],width=.65)
        ax.set_xticks(positions,[f'{r.scenario}:{r.position}' for r in d.itertuples()],rotation=45,ha='right');ax.set(ylim=(0,2480),ylabel='Valid participants',yticks=[0,1000,2000])
        for i,row in enumerate(d.itertuples()):ax.text(i,row.n_participants+45,str(row.n_participants),ha='center',va='bottom',fontsize=6.5)
        self.panel('S_coverage','','Valid participants by clip position',['tables/a2_position_means.csv'],'Twenty-one saved n counts: participants with at least one operated assigned-event rating at each block/clip position. Risk zero only is no operation; pre/post zero is valid.',n='source n ranges from 2142 to 2164 across 21 positions')
        self.save(f,'figure_s_coverage')

    def nonoperation_single(self):
        f=self.figure(120);c=self.table('d2_country_coverage');rate=100*c.n_zero_no_operation/(c.n_valid_risk_ratings+c.n_zero_no_operation);ax=self.ax(f,37,15,137,96)
        ax.hlines(range(29),0,rate,color=BLUE,lw=.9,alpha=.65);ax.scatter(rate,range(29),color=BLUE,s=12)
        ax.set_yticks(range(29),[f'{SHORT.get(r.country,r.country)}  {r.n_completed_licensed}' for r in c.itertuples()]);ax.set(ylim=(28.6,-.7),xlim=(-.15,10.5),xticks=[0,2,4,6,8,10],xlabel='Video-risk responses recorded as zero / no operation (%)')
        for i,val in enumerate(rate):ax.text(val+.10,i,f'{val:.2f}%',va='center',fontsize=6.5)
        self.panel('S_nonoperation','','Video-risk no-operation fractions',['tables/d2_country_coverage.csv'],'Numerator: n_zero_no_operation; denominator: n_valid_risk_ratings + n_zero_no_operation, within each of all 29 countries. Labels include participant n. Repeated risk responses, not independent people; pre/post zeros excluded from this calculation.',transform='100 × saved zero count / saved (valid + zero) counts; equal-size dots, line from zero to each fraction.')
        self.save(f,'figure_s_nonoperation')

    def pair_single(self,feature):
        is_lks=feature=='trust_pre_lks';number='S_age_lks' if is_lks else 'S_age_acc';name='figure_s_age_lks' if is_lks else 'figure_s_age_acc'
        f=self.figure(80,86);pairs=pd.read_parquet(self.source/'prediction/m2_sampled_true_interactions.parquet')
        d=pairs[(pairs.feature_a=='age')&(pairs.feature_b==feature)].groupby('participant_id',as_index=False).agg(value_a=('value_a','first'),value_b=('value_b','first'),pair_contribution=('pair_contribution','mean'))
        rng=np.random.default_rng(20261001);xx=d.value_a.to_numpy();yy=d.value_b.to_numpy()
        if d.value_a.nunique()<15:xx=xx+rng.uniform(-.1,.1,len(xx))
        if d.value_b.nunique()<15:yy=yy+rng.uniform(-.1,.1,len(yy))
        lim=max(d.pair_contribution.abs().quantile(.99),1e-5);ax=self.ax(f,17,15,52,55)
        sc=ax.scatter(xx,yy,c=d.pair_contribution,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),s=12,alpha=.8,lw=0)
        ax.set(xlabel='Age (years)',ylabel='Pre-task '+LABELS[feature],ylim=(-.5,10.5),yticks=[0,2,4,6,8,10]);self.cb(f,sc,(73,15,2,55),'Pair SHAP')
        self.panel(number,'','Age by pre-task '+LABELS[feature],['prediction/m2_sampled_true_interactions.parquet'],'100 held-out participants; 400 sampled event rows, four per person. Each point uses saved true pair contributions averaged over that participant\'s four sampled events.',transform=f'Per-participant averaging, deterministic uniform ±0.1 trust-axis jitter (seed 20261001); age positions unchanged. Symmetric colour limits ±{lim:.17g} are the original 99th absolute percentile; values outside are colour-clipped only.',n=100)
        self.save(f,name)

    def workflow(self):
        b=self;f=b.figure(124,180)
        flow=b.table('d2_sample_flow').set_index('measure')['n'].to_dict()
        required={'completed_licensed':2164,'assigned_events':34624,'events_with_nonzero_risk':34566,'administered_window_slots':181776,'zero_no_operation_windows':1810,'valid_nonzero_risk_ratings':179966}
        assert all(int(flow[k])==v for k,v in required.items())
        assert 2341-177==2164 and 2164*84==181776 and 179966+1810==181776 and 34566+58==34624
        b.text(f,90,73,'a',fontsize=8,fontweight='bold',ha='center',va='center')
        cols=[(5,32,'Recruitment','Prolific\n11 July–19 September 2023'),(45,38,'Preparation','Consent and eligibility\nBackground information\nSix pre-task attitude items\nPractice'),(91,45,'Video task','16 events; four per scenario\nConsecutive 6 s clips\n84 rating opportunities/person\nMost dangerous moment rated\nafter each clip'),(144,31,'Post-task ratings','Six attitude items\n0–10 response scale')]
        for j,(x,w,title,body) in enumerate(cols):
            b.text(f,x,115,title,fontsize=7,fontweight='bold',va='top')
            b.text(f,x,108,body,fontsize=7,va='top',linespacing=1.45)
            f.add_artist(matplotlib.lines.Line2D([x/180,(x+w)/180],[118/124,118/124],transform=f.transFigure,color=INK,lw=.7))
            if j<3:f.add_artist(matplotlib.patches.FancyArrowPatch(((x+w+1)/180,116/124),((cols[j+1][0]-2)/180,116/124),transform=f.transFigure,arrowstyle='->',mutation_scale=6,lw=.55,color=GREY))
        b.text(f,5,81,'Preparation components are grouped; their finer order is not inferred.',fontsize=6.5,va='top',color=GREY)
        b.text(f,90,1,'b',fontsize=8,fontweight='bold',ha='center',va='bottom')
        for y in [62,45,28,11]:f.add_artist(matplotlib.lines.Line2D([5/180,175/180],[y/124,y/124],transform=f.transFigure,color=LIGHT,lw=.6))
        rows=[(55,'Participants','2,341 starters','177 incomplete or <504 s','2,164 retained','−','='),(38,'Clip ratings','181,776 assigned','1,810 no-operation zeros','179,966 operated','−','='),(21,'Person–events','34,624 assigned','58 all-no-operation','34,566 analysed','−','=')]
        for y,label,total,removed,retained,minus,equal in rows:
            for x,t,kw in [(5,label,{'fontweight':'bold'}),(41,total,{}),(80,minus,{}),(90,removed,{}),(139,equal,{}),(147,retained,{})]:b.text(f,x,y,t,fontsize=7,va='center',**kw)
        b.text(f,5,6,'Video-risk zero = no operation; pre- and post-task attitude zero = valid score.',fontsize=6.5,va='center')
        b.panel('FigS1','a','Study procedure',['docs/study_protocol.json'],'Protocol-derived procedure; preparation order not over-specified.')
        b.panel('FigS1','b','Participant and response accounting',['tables/d2_sample_flow.csv'],'Participant and observation denominators checked separately.')
        b.save(f,'figure_s_workflow')
        report={'status':'pass','derived_counts':required,'protocol_counts':{'started':2341,'excluded_incomplete_or_under_504s':177,'retained':2164},'all_accounting_identities_checked':True,'source_sha256':digest(b.source/'tables/d2_sample_flow.csv'),'layout':'Monochrome rules and text; no decorative imagery, gradients or data re-estimation.'}
        (b.reportdir/'workflow_accounting_validation.json').write_text(json.dumps(report,indent=2)+'\n')
        return b.outputs[0]

    def robustness_combined(self):
        f=self.figure(125);d=self.table('a2_sensitivity_profiles')
        variants=[('positive_primary','Primary: operated scores',BLUE,'-'),('zeros_as_scores_diagnostic','Zeros as scores (diagnostic)',GREY,':'),('positive_duration_qc','Duration subset',ORANGE,'--'),('positive_complete_events','Complete events',TEAL,'-.'),('positive_equal_country_weight','Equal country weighting',PURPLE,'--')]
        handles=None
        for i,(letter,block) in enumerate(zip('abcd',['B1','B2','B3','B4'])):
            x=17+(i%2)*89;y=78-(i//2)*47;ax=self.ax(f,x,y,65,33)
            for variant,label,color,ls in variants:
                q=d[(d.scenario==block)&(d.variant==variant)].sort_values('position');ax.plot(q.position,q['mean'],marker='o',ms=2.4,lw=1.2 if variant=='positive_primary' else .8,color=color,ls=ls,label=label)
            ax.set(xlabel='Clip position',ylabel='Mean risk score',ylim=(1,5.5),yticks=[1,2,3,4,5],xticks=range(1,7 if block=='B1' else 6));self.heading(f,letter,block,x-14,y+43)
            handles,labels=ax.get_legend_handles_labels()
            self.panel('S_robustness',letter,block+' sensitivity curves',['tables/a2_sensitivity_profiles.csv'],'Five saved variants: positive-score primary, zeros-as-scores diagnostic, duration-QC subset, complete-event subset, equal-country weighting. No new CI or exclusion rule.',n='varies by saved variant and clip position')
        f.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.002),ncol=3,fontsize=6.5,handlelength=2,columnspacing=1.2,labelspacing=.65)
        self.save(f,'figure_s_robustness')

    def full_importance(self):
        f=self.figure(140);d=self.table('m2_shap_global_importance').sort_values('mean_abs_shap');assert len(d)==26
        ax=self.ax(f,58,16,115,114);ax.hlines(range(26),0,d.mean_abs_shap,color=BLUE,lw=.7,alpha=.55);ax.scatter(d.mean_abs_shap,range(26),s=13,color=BLUE,zorder=3)
        ax.set_yticks(range(26),[LABELS.get(v,v) for v in d.feature]);ax.set_xlabel('Mean absolute SHAP (risk points)');ax.set_ylim(-.7,25.7);ax.xaxis.set_major_locator(MaxNLocator(4))
        self.panel('S_importance','','All model-feature attributions',['tables/m2_shap_global_importance.csv'],'All 26 source variables, including stimulus identifiers and reported country; saved mean absolute TreeSHAP contributions across 34566 held-out event rows. No uncertainty interval.',transform='Full saved rank; same variable aggregation as main Fig. 4a.')
        self.save(f,'figure_s_full_importance')

    def full_pair_matrix(self):
        from itertools import combinations
        f=self.figure(170);d=self.table('m2_global_pair_interactions');features=self.table('m2_shap_global_importance').feature.tolist();assert len(features)==26 and len(d)==325
        expected={frozenset(x) for x in combinations(features,2)};assert {frozenset((r.feature_a,r.feature_b)) for r in d.itertuples()}==expected
        matrix=np.full((26,26),np.nan)
        for r in d.itertuples():
            i,j=features.index(r.feature_a),features.index(r.feature_b);matrix[i,j]=matrix[j,i]=r.mean_abs_pair_contribution
        masked=np.ma.masked_invalid(matrix);ax=self.ax(f,45,39,118,118);mesh=ax.pcolormesh(np.arange(27)-.5,np.arange(27)-.5,masked,cmap='Purples',vmin=0,rasterized=False)
        labels=[LABELS.get(v,v) for v in features];ax.set_ylim(25.5,-.5);ax.set_yticks(range(26),labels);ax.set_xticks(range(26),labels,rotation=55,ha='right');ax.tick_params(length=0,labelsize=6.5,pad=2)
        self.cb(f,mesh,(169,39,1.8,118),'Mean |pair SHAP|');self.panel('S_pairs','','All true pair contributions',['tables/m2_global_pair_interactions.csv','tables/m2_shap_global_importance.csv'],'All 325 unordered pairs among 26 model source variables; symmetric display of mean absolute twice-off-diagonal TreeSHAP interaction from 400 held-out event rows, 100 people. Diagonal masked, not zero-valued estimated main effects.',transform='Symmetric matrix, each saved unordered pair appears twice for alignment; white diagonal is masked.',n=100)
        self.save(f,'figure_s_full_pair_matrix')

    def country_families(self):
        f=self.figure(135);d=self.table('m2_country_attributions');countries=self.table('d2_country_coverage').country.tolist();counts=d.groupby('country').n_participants.first();order=list(FAMILY)
        wide=d.pivot(index='country',columns='family',values='mean_absolute_contribution').reindex(countries);ax=self.ax(f,35,24,138,101);left=np.zeros(29)
        for fam,col in zip(order,FAMILY_COLORS):
            ax.barh(range(29),wide[fam],left=left,color=col,height=.72,label=FAMILY_SHORT[fam]);left+=wide[fam].to_numpy()
        ax.set_yticks(range(29),[f'{SHORT.get(c,c)}  {counts[c]}' for c in countries]);ax.tick_params(axis='y',labelsize=6.5);ax.set(ylim=(28.6,-.7),xlabel='Mean absolute family contribution (risk points)');ax.xaxis.set_major_locator(MaxNLocator(5))
        handles,labels=ax.get_legend_handles_labels();f.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.005),ncol=4,frameon=False,fontsize=6.5,handlelength=1.1,columnspacing=1)
        self.panel('S_country_families','','Family attribution by reported country',['tables/m2_country_attributions.csv','tables/d2_country_coverage.csv'],'Previous main profile-figure country panel: all 29 countries in descending participant count; seven stacked mean absolute signed family sums. Known-country participant-held-out models; not country-held-out explanations. No new uncertainty interval.')
        self.save(f,'figure_s_country_attributions')

    def substitution_and_performance(self):
        f=self.figure(90);d=self.table('m2_age_experience_sensitivity_attribution');full=self.table('m2_shap_global_importance').set_index('feature');ax=self.ax(f,30,28,53,47)
        for j,(spec,feature) in enumerate([('without_age','driving_exp_years'),('without_driving_exp_years','age')]):
            q=d[(d.specification==spec)&(d.feature==feature)];base=float(full.loc[feature,'mean_abs_shap']);removed=float(np.average(q.mean_abs_shap,weights=q.n));ax.plot([base,removed],[j,j],color=LIGHT,lw=1.3);ax.scatter(base,j,s=16,color=BLUE,label='Full model' if j==0 else None);ax.scatter(removed,j,s=18,color=ORANGE,marker='D',label='Other variable removed' if j==0 else None)
        ax.set_yticks(range(2),['Experience\n(age removed)','Age\n(experience removed)']);ax.set(ylim=(1.6,-.6),xlim=(0,.13),xticks=[0,.05,.10],xlabel='Mean absolute SHAP (risk points)');ax.legend(loc='lower left',bbox_to_anchor=(-.20,-.45),fontsize=6.5,handletextpad=.4);self.heading(f,'a','Attribution substitution',3,87)
        self.panel('S_substitution','a','Correlated-attribute attribution substitution',['tables/m2_age_experience_sensitivity_attribution.csv','tables/m2_shap_global_importance.csv'],'Previous main profile-figure attribution substitution: saved fold-specific mean absolute SHAP weighted by event count; identical saved refits, no new fitting.')
        perf=self.table('m2_performance');base=perf[(perf.protocol=='participant_5fold')&(perf.model=='LightGBM')].iloc[0];ab=self.table('m2_age_experience_sensitivity_metrics').set_index('model');rows=[base,ab.loc['LightGBM_without_age'],ab.loc['LightGBM_without_driving_exp_years']];ax=self.ax(f,126,28,48,47)
        for i,(row,col) in enumerate(zip(rows,[BLUE,ORANGE,TEAL])):ax.errorbar(row.r2,i,xerr=[[row.r2-row.r2_ci_low],[row.r2_ci_high-row.r2]],fmt='o',ms=3,color=col,lw=.8,capsize=2)
        ax.set(yticks=range(3),yticklabels=['Full model','Without age','Without experience'],ylim=(2.6,-.6),xlabel='Held-out R²',xlim=(.20,.245),xticks=[.20,.22,.24]);self.heading(f,'b','Held-out performance',96,87)
        self.panel('S_substitution','b','Held-out performance after feature removal',['tables/m2_performance.csv','tables/m2_age_experience_sensitivity_metrics.csv'],'Saved LightGBM participant-held-out performance: full, age omitted, or driving-experience omitted. 34566 event means; no additional tuning or refitting. Intervals are marginal per-model intervals, not a CI for the difference.','Saved pointwise 95% participant-cluster bootstrap intervals; fixed held-out predictions.')
        self.save(f,'figure_s_attribute_substitution')

    def country_holdout_rmse(self):
        f=self.figure(155);d=self.table('m2_country_holdout_performance');coverage=self.table('d2_country_coverage');countries=coverage.country.tolist();models=['Training event/block mean','Training mean','Ridge','LightGBM','Ridge identity only','LightGBM identity only'];wide=d.pivot(index='country',columns='model',values='rmse').reindex(index=countries,columns=models);assert wide.shape==(29,6) and not wide.isna().any().any()
        ax=self.ax(f,35,25,131,120);mesh=ax.pcolormesh(np.arange(7)-.5,np.arange(30)-.5,wide.to_numpy(),cmap='Blues',vmin=0,rasterized=False)
        for i in range(29):
            for j in range(6):ax.text(j,i,f'{wide.iloc[i,j]:.2f}',ha='center',va='center',fontsize=6.5,color='white' if wide.iloc[i,j]>.6*wide.to_numpy().max() else INK)
        ax.set_ylim(28.5,-.5);ax.set_yticks(range(29),[f'{SHORT.get(r.country,r.country)}  {r.n_completed_licensed}' for r in coverage.itertuples()]);ax.set_xticks(range(6),['Event/block\nmean','Grand\nmean','Ridge','LightGBM','Ridge\nidentity only','LightGBM\nidentity only']);ax.tick_params(length=0,labelsize=6.5,pad=3);self.cb(f,mesh,(171,25,1.8,120),'RMSE')
        self.panel('S_country_holdout','','Country-held-out RMSE by model',['tables/m2_country_holdout_performance.csv','tables/d2_country_coverage.csv'],'All six saved models and all 29 countries; numeric cells show RMSE in risk-score points rounded to two decimals. Country labels include participant n. No country-specific confidence intervals; small-country point estimates can be unstable.',transform='Fixed saved RMSE table; no weighting change, fitting or uncertainty calculation.')
        self.save(f,'figure_s_country_holdout_rmse')

    def event_completeness(self):
        f=self.figure(125);d=self.table('d2_event_window_completeness')
        for i,(letter,block) in enumerate(zip('abcd',['B1','B2','B3','B4'])):
            x=17+(i%2)*89;y=73-(i//2)*55;ax=self.ax(f,x,y,65,36);q=d[d.scenario==block];assert q.n_events.sum()==8656
            ax.bar(q.n_valid_windows,q.n_events,color=COLORS[i],width=.66);ax.set_yscale('log');ax.set(xlabel='Operated clips in an assigned event',ylabel='Event sequences (log scale)',ylim=(1,18000),xticks=range(0,7 if block=='B1' else 6),yticks=[1,10,100,1000,10000]);ax.yaxis.set_major_formatter(FormatStrFormatter('%g'));ax.minorticks_off()
            for r in q.itertuples():ax.text(r.n_valid_windows,r.n_events*1.2,str(r.n_events),ha='center',va='bottom',fontsize=6.5)
            self.heading(f,letter,block,x-14,y+47);self.panel('S_completeness',letter,block+' event completeness',['tables/d2_event_window_completeness.csv'],'Counts of assigned participant–event sequences by number of operated clips; each block has 8656 assigned sequences. B1 has six administered clips; the other blocks have five. Zero here means no operated clip in that assigned event.',transform='Logarithmic count axis is labelled; no zeros are plotted as risk scores.')
        self.save(f,'figure_s_event_completeness')

    def remaining_dependence(self,feature):
        short={'trust_pre_overall':'overall_trust','accept_pre_delegate':'delegation','accept_pre_monitor_raw':'supervision','accept_pre_distract':'other_activities'}[feature];f=self.figure(80,86);d=pd.read_parquet(self.source/'prediction/m2_oof_shap.parquet').sample(6000,random_state=20261001);assert d.participant_id.nunique()==2057
        ax=self.ax(f,18,17,63,55);ax.scatter(d['value__'+feature],d['shap__'+feature],s=3,alpha=.13,color=BLUE,lw=0,rasterized=False);ax.axhline(0,color=GREY,lw=.5);ax.set(xlabel='Pre-task '+LABELS[feature].lower(),ylabel='SHAP contribution (risk points)',xticks=[0,2,4,6,8,10]);ax.yaxis.set_major_locator(MaxNLocator(4))
        self.panel('S_shap_'+short,'',LABELS[feature]+' SHAP dependence',['prediction/m2_oof_shap.parquet'],'Same fixed 6000 held-out event-row sample from 2057 people as main Fig. 3b–f. Numeric pre-task score versus its saved TreeSHAP contribution, not an intervention effect. A person can contribute multiple rows.',transform='Original event rows without jitter, resampling or new exclusions.',n=2057)
        self.save(f,'figure_s_shap_'+short)

    def whatif_main_panels(self,f):
        order=['trust_pre_overall','trust_pre_acc','trust_pre_lks','accept_pre_delegate','accept_pre_monitor_raw','accept_pre_distract']
        d=pd.read_csv(self.input('whatif_primary.csv'));d=d[(d.scope=='pooled')&(d.min_count==10)];assert len(d)==12 and d.status.eq('estimated').all();ax=self.ax(f,37,23,60,49)
        short=['Overall trust','ACC trust','LKS trust','Delegation','Supervision need','Other activities'];labels=[]
        for i,feature in enumerate(order):
            q=d[d.context==feature].set_index('contrast');assert set(q.index)=={-1,1} and q.n_eligible.nunique()==1
            labels.append(f'{short[i]}  {int(q.n_eligible.iloc[0])}')
            for step,col,offset in [(-1,BLUE,-.16),(1,ORANGE,.16)]:
                r=q.loc[step];yy=i+offset;ax.hlines(yy,r.simultaneous_ci_low,r.simultaneous_ci_high,color=col,lw=.6);ax.hlines(yy,r.ci_low,r.ci_high,color=col,lw=1.8);ax.scatter(r.estimate,yy,s=12,color=col,zorder=3,label=f'Score {step:+d}' if i==0 else None)
        ax.axvline(0,color=GREY,lw=.55);ax.set(yticks=range(6),yticklabels=labels,ylim=(5.6,-.6),xlabel='Change in predicted risk (points)');ax.xaxis.set_major_locator(MaxNLocator(4));ax.tick_params(axis='y',labelsize=6.5)
        ax.legend(loc='lower left',bbox_to_anchor=(-.08,-.38),ncol=2,fontsize=6.5,handletextpad=.35,columnspacing=1)
        self.heading(f,'e','One-point input changes',3,87);self.text(f,37,78,'Labels include support n',fontsize=6.5)
        self.panel(5,'e','One-point what-if input changes',['saved/whatif_primary.csv'],'Six original pre-task items; fixed held-out models evaluated after a one-point input decrement or increment within each item\'s common eligible support set. Labels give eligible participant n. Model responses, not causal effects.',ci='Thick: saved pointwise 95% participant-bootstrap intervals. Thin: saved simultaneous intervals for the complete 12-comparison primary family. Models are fixed.',n='item-specific common support: 830, 952, 972, 783, 684, 532')
        d=pd.read_csv(self.input('whatif_paired_feature.csv'));d=d[(d.step==1)&(d.feature_contrast=='LKS_minus_ACC')].set_index('scope').reindex(['pooled','B1','B2','B3','B4']);assert len(d)==5 and d.n_eligible.eq(571).all();ax=self.ax(f,130,23,44,49)
        for i,r in enumerate(d.itertuples()):ax.errorbar(r.estimate,i,xerr=[[r.estimate-r.ci_low],[r.ci_high-r.estimate]],fmt='D' if i==0 else 'o',ms=3.3 if i==0 else 3,color=PURPLE,lw=.8,capsize=1.5)
        ax.axvline(0,color=GREY,lw=.55);ax.set(yticks=range(5),yticklabels=['Pooled','B1','B2','B3','B4'],ylim=(4.6,-.6),xlabel='LKS − ACC change\n(risk points)');ax.xaxis.set_major_locator(MaxNLocator(3));self.heading(f,'f','Matched what-if contrast',108,87);self.text(f,130,78,'Common support: n = 571',fontsize=6.5)
        self.panel(5,'f','Matched one-point what-if contrast',['saved/whatif_paired_feature.csv'],'Post-primary exploratory paired difference between LKS +1 and ACC +1 fixed-model risk responses in the same 571 eligible people, pooled and by block; diamond denotes pooled estimate. Both changes refer to the same participant and fixed model.',ci='Saved pointwise 95% paired participant-bootstrap intervals; no simultaneous claim.',n=571)

    def whatif_dose(self):
        f=self.figure(90);d=pd.read_csv(self.input('whatif_dose.csv'));d=d[d.scope=='pooled']
        for j,(feature,label,col) in enumerate([('trust_pre_acc','ACC',ORANGE),('trust_pre_lks','LKS',TEAL)]):
            q=d[d.context==feature].sort_values('contrast');assert len(q)==5 and q.n_eligible.nunique()==1;ax=self.ax(f,18+89*j,20,64,54)
            ax.axhline(0,color=GREY,lw=.55);ax.errorbar(q.contrast,q.estimate,yerr=[q.estimate-q.ci_low,q.ci_high-q.estimate],fmt='o-',color=col,ms=3,lw=.8,capsize=2)
            ax.set(xlabel='Change to pre-task score (points)',ylabel='Change in predicted risk (points)',xticks=[-2,-1,0,1,2]);ax.yaxis.set_major_locator(MaxNLocator(4));self.heading(f,'ab'[j],f'{label}: common support n = {int(q.n_eligible.iloc[0])}',4+89*j,87)
            self.panel('S_whatif_dose','ab'[j],label+' model-response dose profile',['saved/whatif_dose.csv'],'Fixed-model responses to score changes −2, −1, 0, +1, +2 on the common support set for all five settings. Joining segments connect discrete evaluated settings and do not establish a smooth dose-response relationship.','Saved pointwise 95% participant-bootstrap intervals.',n=int(q.n_eligible.iloc[0]))
        self.save(f,'figure_s_whatif_dose')

    def whatif_support(self):
        f=self.figure(95);order=['trust_pre_overall','trust_pre_acc','trust_pre_lks','accept_pre_delegate','accept_pre_monitor_raw','accept_pre_distract'];d=pd.read_csv(self.input('whatif_support_coverage.csv'));d=d[(d.analysis=='primary')&(d.min_count==10)&(d.country=='ALL')].set_index('context').reindex(order);assert len(d)==6
        keys=['n_boundary_excluded','n_radius_excluded_after_boundary','n_count_excluded_after_boundary_radius','n_eligible'];labels=['Outside score boundary','Outside neighbour radius','Insufficient local neighbours','Eligible common support'];colors=[LIGHT,'#F1C05B',BLUE,TEAL];values=d[keys].to_numpy(float);assert np.isfinite(values).all() and np.all(values.sum(1)==2164)
        ax=self.ax(f,41,26,131,57);left=np.zeros(6)
        for j,(key,label,col) in enumerate(zip(keys,labels,colors)):
            ax.barh(range(6),d[key],left=left,color=col,height=.62,label=label);left+=d[key].to_numpy()
        for i,n in enumerate(d.n_eligible):ax.text(2164-n/2,i,f'n = {int(n)}',ha='center',va='center',fontsize=7,color='white')
        ax.set(yticks=range(6),yticklabels=[LABELS[x] for x in order],ylim=(5.6,-.6),xlim=(0,2200),xticks=[0,500,1000,1500,2000],xlabel='Participants');ax.legend(loc='lower center',bbox_to_anchor=(.42,-.44),ncol=2,fontsize=6.5,handlelength=1.2,columnspacing=1)
        self.panel('S_whatif_support','','Support coverage for primary one-point what-if comparisons',['saved/whatif_support_coverage.csv'],'Six one-item common support populations at minimum-neighbour count 10. Exclusions are sequential and mutually exclusive: boundary, then radius among boundary-valid people, then neighbour count among boundary-and-radius-valid people. Each stack totals 2164. Marginal radius/local-count validity fields are not treated as a sequential funnel.',transform='Exact source counts; no new support rule or eligibility filtering.')
        self.save(f,'figure_s_whatif_support')

    def whatif_sensitivity(self):
        f=self.figure(145);order=['trust_pre_overall','trust_pre_acc','trust_pre_lks','accept_pre_delegate','accept_pre_monitor_raw','accept_pre_distract'];d=pd.concat([pd.read_csv(self.input('whatif_support_sensitivity.csv')),pd.read_csv(self.input('whatif_primary.csv'))],ignore_index=True);d=d[(d.scope=='pooled')&d.min_count.isin([5,10,20])];lo=float(d.ci_low.min());hi=float(d.ci_high.max());margin=.12*(hi-lo)
        for i,feature in enumerate(order):
            ax=self.ax(f,17+59*(i%3),91-61*(i//3),36,38);q=d[d.context==feature];assert len(q)==6
            for step,col in [(-1,BLUE),(1,ORANGE)]:
                ss=q[q.contrast==step].sort_values('min_count');ax.errorbar(ss.min_count,ss.estimate,yerr=[ss.estimate-ss.ci_low,ss.ci_high-ss.estimate],fmt='o-',ms=2.5,color=col,lw=.75,capsize=1.5,label=f'Score {step:+d}')
            for threshold in sorted(q.loc[q.estimate.isna(),'min_count'].unique()):
                assert q[q.min_count==threshold].estimate.isna().all()
                ax.annotate('Not\nestimated',xy=(threshold,.78),xycoords=('data','axes fraction'),ha='right',va='center',fontsize=6,color=GREY)
            counts=q[q.contrast==1].set_index('min_count').n_eligible;ax.set(xticks=[5,10,20],xticklabels=[f'{v}\nn={int(counts.loc[v])}' for v in [5,10,20]],ylim=(lo-margin,hi+margin),xlabel='Minimum neighbours');ax.axhline(0,color=GREY,lw=.5);ax.tick_params(axis='x',labelsize=6);ax.yaxis.set_major_locator(MaxNLocator(3))
            if i%3==0:ax.set_ylabel('Predicted-risk change')
            self.heading(f,'abcdef'[i],LABELS[feature],3+59*(i%3),141-61*(i//3));self.panel('S_whatif_sensitivity','abcdef'[i],LABELS[feature]+' support-threshold sensitivity',['saved/whatif_primary.csv','saved/whatif_support_sensitivity.csv'],'Pooled fixed-model responses to one-point decrements/increments under minimum-neighbour counts 5, 10, 20. Eligible participants differ by threshold; axis labels report n. Source rows with insufficient support remain unestimated and are explicitly labelled; no zero effect is imputed. This changes the support population, not only a numerical tuning value.','Saved pointwise 95% participant-bootstrap intervals.',n='threshold- and item-specific; labelled in the plot')
        handles,labels=ax.get_legend_handles_labels();f.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.006),ncol=2,fontsize=6.5,handlelength=1.5)
        self.save(f,'figure_s_whatif_sensitivity')

    def parameter_adjusted(self,scenario):
        f=self.figure(72);d=pd.read_csv(self.input('controlled_parameter_adjusted_sensitivity.csv'));d=d[(d.analysis_variant=='assigned_design_primary')&(d.display_scenario==scenario)];assert d.status.eq('estimated').all()
        distances={'LC':'Neighbour merging distance (m)','SVM':'Ego merging distance (m)','HB':'Initial braking distance (m)','MB':'Merging distance (m)'}
        factors=[('design_distance',distances[scenario]),('lateral_behaviour','Lateral behaviour'),('acc_style','ACC style')] if scenario=='LC' else [('design_distance',distances[scenario]),('design_speed','Cruising speed (km/h)'),('design_braking','Braking (m/s²)')]
        seen=0
        for j,(factor,title) in enumerate(factors):
            q=d[d.factor==factor];ax=self.ax(f,16+60*j,15,40,35);seen+=len(q);ax.axhline(0,color=GREY,lw=.6)
            comparisons=q[['comparison_level','comparison_label','reference_label']].drop_duplicates()
            for (level,comparison,reference),color in zip(comparisons.itertuples(index=False,name=None),COLORS):
                z=q[q.comparison_level==level].sort_values('clip');label=(str(comparison)+' vs '+str(reference)).replace('-','−')
                ax.fill_between(z['clip'],z.ci_low,z.ci_high,color=color,alpha=.15,lw=0);ax.plot(z['clip'],z.estimate,'o-',ms=2.6,color=color,lw=.9,label=label)
            ax.set(xlabel='Clip position',xticks=range(1,7 if scenario=='LC' else 6),ylim=(-4.1,1.8),yticks=[-4,-2,0]);
            if j==0:ax.set_ylabel('Adjusted risk difference (points)')
            ax.legend(loc='lower center',bbox_to_anchor=(.5,1.06),ncol=1,fontsize=6,handlelength=1.1,handletextpad=.3,borderaxespad=0,labelspacing=.15)
            self.heading(f,'abc'[j],scenario+' · '+title,3+60*j,70)
            self.panel('S_adjusted_'+scenario,'abc'[j],scenario+' adjusted '+title,['saved/controlled_parameter_adjusted_sensitivity.csv'],'Assigned-design primary subset; all non-reference categorical factor coefficients from the saved additive weighted participant fixed-effects model at each clip. Curves are comparison minus reference; both levels are printed in the legend. Other included design factors are adjusted; this is a sensitivity analysis, not a causal estimate.','Saved pointwise 95% participant-bootstrap coefficient intervals, 2000 valid draws; no new fitting or resampling.',n='clip-specific, given in source')
        assert seen==len(d) and len(d)==(36 if scenario=='LC' else 30)
        self.save(f,'figure_s_parameters_adjusted_'+scenario.lower())

    def lc_distance_sensitivity(self):
        f=self.figure(72);d=pd.read_csv(self.input('controlled_parameter_clip_profiles.csv'));d=d[(d.display_scenario=='LC')&(d.factor=='design_distance')];assert len(d)==24
        for j,(variant,title) in enumerate([('assigned_design_primary','Assigned design'),('lc_exclude_distance_conflicts','Exclude conflict events')]):
            q=d[d.analysis_variant==variant];assert len(q)==12;ax=self.ax(f,18+89*j,10,65,40)
            for (order,label),color in zip(q[['level_order','level_label']].drop_duplicates().sort_values('level_order').itertuples(index=False,name=None),COLORS):
                z=q[q.level_order==order].sort_values('clip');ax.fill_between(z['clip'],z.ci_low,z.ci_high,color=color,alpha=.15,lw=0);ax.plot(z['clip'],z['mean'],'o-',ms=2.7,lw=.9,color=color,label=label)
            ax.set(xlabel='Clip position',ylabel='Mean risk',xticks=range(1,7),ylim=(1,8),yticks=[1,3,5,7]);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,fontsize=6.5,handlelength=1.2,borderaxespad=0)
            self.heading(f,'ab'[j],'LC · '+title,4+89*j,69)
            self.panel('S_lc_distance_sensitivity','ab'[j],'LC neighbour merging distance: '+title,['saved/controlled_parameter_clip_profiles.csv'],'Saved '+variant+' distance profiles; levels 5 m and 15 m. The diagnostic excludes physically inconsistent distance events and changes the observed event/support population. Neither relabels the excluded events nor reassigns their risk scores.','Saved pointwise 95% participant-bootstrap mean intervals, 2000 draws.',n=f"varies from {int(q.n_participants.min())} to {int(q.n_participants.max())}")
        self.save(f,'figure_s_lc_distance_sensitivity')

    def dataset(self):
        f=self.figure(180,180);c=self.table('d2_country_coverage');n=c.n_completed_licensed.to_numpy()
        ax=self.ax(f,29,110,65,59);col=[TEAL if z>=30 else GREY for z in n];ax.hlines(range(29),0,n,color=col,lw=.75,alpha=.55);ax.scatter(n,range(29),color=col,s=12,zorder=3)
        ax.set_yticks(range(29),[SHORT.get(cn,cn) for cn in c.country]);ax.set(ylim=(28.6,-.7),xlim=(-20,660),xticks=[0,200,400,600],xlabel='')
        for y,z in enumerate(n):ax.annotate(str(z),(z,y),xytext=(3.5,0),textcoords='offset points',va='center',fontsize=6)
        ax.tick_params(axis='y',labelsize=6)
        self.heading(f,'a','Reported countries (n) and risk scores',3,178)
        self.panel(1,'a','Reported countries',['tables/d2_country_coverage.csv'],'All 29 countries retained; Grey marks flag n < 30 only; lines connect zero to the exact participant count.')
        ax=self.ax(f,59,119,33,30);d=self.table('d2_risk_score_distribution').query("scenario == 'All'")
        col=[TEAL if k<=3 else BLUE for k in d.risk_score];ax.vlines(d.risk_score,0,d.proportion_valid_ratings*100,color=col,lw=1.1,alpha=.7);ax.scatter(d.risk_score,d.proportion_valid_ratings*100,c=col,s=8,zorder=3)
        ax.set(xticks=[1,3,5,7,9],yticks=[0,20,40],xlabel='Risk score',ylabel='Ratings (%)',ylim=(0,42));ax.tick_params(labelsize=6,pad=1)
        ax.xaxis.label.set_fontsize(6);ax.yaxis.label.set_fontsize(6);ax.yaxis.labelpad=1
        self.text(f,58,154,'179,966 valid ratings',fontsize=6)
        self.panel(1,'a-inset','Risk-score distribution',['tables/d2_risk_score_distribution.csv'],'Inset in panel a: 179966 repeated valid scores 1-10; 1810 recorded zeros denote no operation and are omitted. Integer-score stems have no smoothing; scores 1–3 are highlighted descriptively.')
        p=pd.read_parquet(self.data/'participants.parquet');ax=self.ax(f,125,137,38,38)
        h=ax.hexbin(p.age,p.driving_exp_years,gridsize=24,mincnt=1,cmap='Blues',linewidths=.15,rasterized=False)
        ax.set(xlabel='Age (years)',ylabel='Driving experience (years)');self.cb(f,h,(169,137,1.6,38),'People')
        self.heading(f,'b','Age and driving experience',103,178)
        self.panel(1,'b','Age and driving experience',['data/participants.parquet'],'Original age and driving-experience values, one row per participant.',transform='Hexbin gridsize 24, unchanged from previous display.')
        ax=self.ax(f,125,112,45,16);g=self.table('d2_gender_risk_descriptive');ax.errorbar(g.risk_mean,range(3),xerr=[g.risk_mean-g.ci_low,g.ci_high-g.risk_mean],fmt='o',ms=3,color=TEAL,capsize=2,lw=.8)
        ax.set_yticks(range(3),[f'{r.gender} (n={r.n:,})' for r in g.itertuples()]);ax.set(ylim=(2.5,-.5),xlim=(2.4,3.7),xticks=[2.5,3,3.5],xlabel='Mean operated risk')
        ax.tick_params(axis='y',labelsize=6,pad=2)
        ax.xaxis.label.set_fontsize(6);ax.tick_params(axis='x',labelsize=6)
        self.heading(f,'c','Risk by reported gender',103,130)
        self.panel(1,'c','Risk by reported gender',['tables/d2_gender_risk_descriptive.csv'],'Participant block-balanced risk; unadjusted; not-reported group retained.','Pointwise 95% participant bootstrap, 1000 draws.')
        ph=pd.read_csv(self.input('prepost_dispersion.csv'))
        d=pd.DataFrame([{'label':r.label,'metric':metric,'mean':getattr(r,metric+'_mean'),'ci_low':getattr(r,metric+'_ci_low'),'ci_high':getattr(r,metric+'_ci_high'),'sd':getattr(r,metric+'_sd')} for r in ph.itertuples() for metric in ['pre','post']])
        labels=d.label.drop_duplicates().tolist();ax=self.ax(f,33,73,48,28)
        short={'Overall trust':'Overall trust','ACC trust':'ACC trust','LKS trust':'LKS trust','Delegate driving':'Delegation','Need for supervision':'Supervision need','Non-driving activities':'Non-driving activities'}
        for i,label in enumerate(labels):
            q=d[(d.label==label)&d.metric.isin(['pre','post'])].set_index('metric');ax.plot(q.loc[['pre','post'],'mean'],[i,i],color=LIGHT,lw=1.2)
            for metric,col,offset in [('pre',BLUE,-.16),('post',TEAL,.16)]:
                r=q.loc[metric];ax.hlines(i+offset,r['mean']-r.sd,r['mean']+r.sd,color=col,lw=.65,alpha=.50);ax.errorbar(r['mean'],i+offset,xerr=[[r['mean']-r.ci_low],[r.ci_high-r['mean']]],fmt='o',color=col,ms=2.7,capsize=1,lw=.6,label=metric.capitalize() if i==0 else None)
        ax.set_yticks(range(6),[short[x] for x in labels]);ax.set(xlim=(0,10.5),ylim=(5.6,-.6),xticks=[0,5,10],xlabel='Score (0–10)')
        ax.legend(loc='upper left',bbox_to_anchor=(-.05,1.13),ncol=2,handletextpad=.3,columnspacing=.7)
        self.heading(f,'d','Trust and acceptance',3,102)
        self.panel(1,'d','Trust and acceptance',['saved/prepost_dispersion.csv'],'Six separate original items; zero valid; supervision retained in raw direction. Harmonized saved analysis SEED=20261012; no re-estimation by figure builder.','Thin pale horizontal bars = mean ± sample SD (ddof=1); thicker pointwise 95% participant-bootstrap mean intervals reuse 2000 draws. SD intervals can exceed the 0–10 item range and are not quantiles.')
        for letter,item,x in [('e','acc',89),('f','lks',136)]:
            self.sankey(self.ax(f,x,70,38,31),item);self.heading(f,letter,f'{item.upper()} transitions',x,102)
            self.panel(1,letter,f'{item.upper()} score transitions',['tables/d2_item_transitions_fixed_bands.csv'],'2164 paired participants; display bands 0-3, 4-6, 7-10 are arbitrary descriptive bins, not validated categories.')
        fields=['age','driving_exp_years','risk_person_block_balanced_mean']+NUMERIC[2:]
        labels=['Age','Experience','Mean risk','Overall','ACC','LKS','Delegate','Supervise','Non-driving']
        cor=self.table('d2_spearman_associations');mat=np.eye(9);qmat=np.ones((9,9))
        for r in cor.itertuples():i,j=fields.index(r.variable_a),fields.index(r.variable_b);mat[i,j]=mat[j,i]=r.rho;qmat[i,j]=qmat[j,i]=r.q_bh
        mask=np.triu(np.ones((8,8),bool),1);z=np.ma.array(mat[1:,:8],mask=mask);ax=self.ax(f,28,13,49,49)
        mesh=ax.pcolormesh(np.arange(9)-.5,np.arange(9)-.5,z,cmap='RdBu_r',vmin=-1,vmax=1,rasterized=False)
        for i in range(8):
            for j in range(i+1):ax.text(j,i,f'{mat[i+1,j]:.2f}'.replace('-0.00','0.00'),ha='center',va='center',fontsize=6,color='white' if abs(mat[i+1,j])>.6 else INK)
        ax.set(ylim=(7.5,-.5));ax.set_xticks(range(8),labels[:-1],rotation=50,ha='right');ax.set_yticks(range(8),labels[1:]);ax.tick_params(length=0)
        self.cb(f,mesh,(83,13,1.8,49),'Spearman ρ',ticks=[-1,0,1]);self.heading(f,'g','Participant-level associations',3,66)
        self.panel(1,'g','Participant-level associations',['tables/d2_spearman_associations.csv'],'Saved Spearman correlations for 36 unique off-diagonal pairs; self-correlations omitted. Risk is participant block-balanced mean. Exact n/p/BH q remain in source.',transform='Strict off-diagonal triangle compacted to eight rows by eight columns; all 36 original saved pairwise rho values retained. No significance marks. Rounded negative-zero labels are displayed as 0.00; saved rho and colour values are unchanged.')
        ax=self.ax(f,107,13,66,43);source=self.input('figure_1h_hb_distance_boxstats.csv');boxes=pd.read_csv(source)
        assert len(boxes)==15 and set(boxes.level)=={5,15,25}
        for level,colour,offset in zip([5,15,25],[BLUE,ORANGE,TEAL],[-.24,0,.24]):
            q=boxes[boxes.level==level].sort_values('clip');assert list(q['clip'])==[1,2,3,4,5]
            stats=[{'q1':r.q1,'med':r.median,'q3':r.q3,'whislo':r.whisker_low,'whishi':r.whisker_high,'fliers':json.loads(r.fliers_json)} for r in q.itertuples()]
            ax.bxp(stats,positions=q['clip'].to_numpy()+offset,widths=.20,showfliers=True,showmeans=False,patch_artist=True,manage_ticks=False,
                   boxprops={'facecolor':colour,'edgecolor':colour,'linewidth':.65,'alpha':.25},
                   medianprops={'color':colour,'linewidth':1.0},whiskerprops={'color':colour,'linewidth':.65},capprops={'color':colour,'linewidth':.65},
                   flierprops={'marker':'o','markersize':1.0,'markeredgewidth':0,'markerfacecolor':colour,'alpha':.20})
        ax.set(xlim=(.48,5.52),ylim=(.7,13.25),xticks=[1,2,3,4,5],yticks=[1,3,5,7,9],xlabel='Clip position',ylabel='Participant mean risk')
        handles=[Rectangle((0,0),1,1,facecolor=c,edgecolor=c,alpha=.45,label=f'{v} m') for v,c in zip([5,15,25],[BLUE,ORANGE,TEAL])]
        ax.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,1.005),ncol=3,columnspacing=1.2,handlelength=1.2,handletextpad=.5,borderaxespad=0)
        tests=pd.read_csv(self.input('figure_1h_hb_distance_paired_tests.csv'))
        positions={5:-.24,15:0,25:.24}
        for r in tests.itertuples():
            rank={(15,5):0,(25,15):1,(25,5):2}[(r.comparison_level,r.reference_level)]
            y=10.45+rank*.84;x1=r.clip+positions[r.reference_level];x2=r.clip+positions[r.comparison_level]
            ax.plot([x1,x1,x2,x2],[y,y+.12,y+.12,y],color=INK,lw=.45,clip_on=False)
            ax.text((x1+x2)/2,y+.15,r.significance,ha='center',va='bottom',fontsize=6,color=INK)
        self.heading(f,'h','HB initial braking distance',99,65)
        self.panel(1,'h','HB initial braking distance: participant distributions',[
            'saved/figure_1h_hb_distance_boxstats.csv',
            'saved/figure_1h_hb_distance_paired_tests.csv'],
            'Fifteen boxes cross five clips and three initial braking distances (5/15/25 m). Each point underlying a box is the available-event mean within person, distance and clip. Brackets show paired participant-mean differences; Holm adjustment spans all 195 pairwise factor-level/clip tests across all four scenarios. This panel contains 15 of that complete family. Selection used physical interpretability, coverage and temporal structure, before computing the new p-values.',
            'Median, linear quartiles, most extreme observed values within 1.5 IQR, and all fliers. These are descriptive spread, not confidence intervals.',
            n=f'varies from {int(boxes.n_participants.min())} to {int(boxes.n_participants.max())}')
        self.save(f,'figure_1_dataset')

    def parameter_boxes(self,scenario):
        with plt.rc_context():
            plt.rcdefaults()
            plt.rcParams.update({'font.family':self.font,'font.size':6.5,'axes.labelsize':6.5,'axes.linewidth':.55,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'xtick.major.size':2,'ytick.major.size':2,'xtick.major.width':.5,'ytick.major.width':.5,'axes.spines.top':False,'axes.spines.right':False,'legend.fontsize':6,'legend.title_fontsize':6.5,'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':300,'figure.facecolor':'white'})
            from .parameter_boxes import draw_factor_panel, FACTORS
            b=pd.read_csv(self.input('parameter_boxstats.csv'));t=pd.read_csv(self.input('parameter_paired_tests.csv'))
            f=self.figure(102,180)
            for i,factor in enumerate(FACTORS[scenario]):
                ax=self.ax(f,10+60*i,17,48,61)
                detail=draw_factor_panel(ax,b,t,scenario,factor)
                ax.set_ylim(.7,15.8 if scenario=='LC' else 13.2)
                self.text(f,34+60*i,3.5,chr(97+i),fontsize=8,fontweight='bold',ha='center',va='bottom')
                self.panel(self.current['id'],chr(97+i),scenario+' '+factor,
                    ['saved/parameter_boxstats.csv','saved/parameter_paired_tests.csv'],
                    json.dumps(detail),ci='Descriptive IQR/observed 1.5-IQR whiskers; annotation Holm family=195; no confidence interval drawn.',
                    n=f"{detail['n_participants_min']}–{detail['n_participants_max']}")
            self.save(f,'parameter_boxes')

