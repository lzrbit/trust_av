"""Portable paths, rendering style and run-local provenance for manuscript plots."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter
from pypdf import PdfReader

REPO=Path(__file__).resolve().parents[1]
INK='#25333B'
SCENARIOS={'B1':'LC','B2':'SVM','B3':'HB','B4':'MB'}
def display_label(value):
    text=re.sub(r'\bB[1-4]\b',lambda m:SCENARIOS[m.group()],str(value))
    for old,new in [('blocks','scenarios'),('block','scenario'),('Blocks','Scenarios'),('Block','Scenario')]:
        text=re.sub(r'\b'+old+r'\b',new,text)
    return text

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def saved_relative(name):
    """Route saved estimates to the documented analysis stage that creates them."""
    if name.startswith(('parameter_', 'figure_1h_hb_distance_')):return Path('outputs/parameter_inference/tables')/name
    if name.startswith('whatif_'):return Path('outputs/whatif/tables')/name
    if name.startswith(('controlled_parameter_','within_person_trust_')):return Path('outputs/parameters/tables')/name
    if name=='prepost_harmonized.csv':return Path('outputs/prepost/tables')/name
    if name.startswith('hb_braking_box'):return Path('outputs/boxes/tables')/name
    if name=='hb_braking_paired_tests.csv':return Path('outputs/hb_tests/tables')/name
    return Path('outputs/presentation/tables')/name

class RenderBase:
    def __init__(self,workspace,data_dir=None,font='Arial',font_dir=None):
        self.root=Path(workspace).resolve();self.source=self.root/'outputs/core'
        self.data=Path(data_dir).resolve() if data_dir else self.root/'inputs/derived'
        self.dest=self.root/'outputs/presentation';self.figdir=self.dest/'figures';self.reportdir=self.dest/'reports'
        for p in [self.figdir,self.reportdir,self.dest/'tables']:p.mkdir(parents=True,exist_ok=True)
        self.panels=[];self.outputs=[];self.layouts=[];self.fonts=[];self.current=None
        if font_dir:
            for p in sorted(Path(font_dir).glob('*.ttf')):
                font_manager.fontManager.addfont(str(p));self.fonts.append({'file':p.name,'sha256':digest(p)})
        selected=font_manager.findfont(font_manager.FontProperties(family=font),fallback_to_default=False)
        self.layout_config=json.loads((REPO/'config/figure_layout.json').read_text())
        self.font=font;self.fonts.append({'selected_font':Path(selected).name,'sha256':digest(selected)})
        plt.rcParams.update({'font.family':font,'font.sans-serif':[font],'font.size':7,
            'axes.labelsize':7,'axes.titlesize':7,'axes.linewidth':.55,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
            'xtick.major.size':2,'ytick.major.size':2,'xtick.major.width':.5,'ytick.major.width':.5,
            'axes.spines.top':False,'axes.spines.right':False,'legend.fontsize':6.5,'legend.frameon':False,
            'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
            'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':300,
            'figure.facecolor':'white','mathtext.fontset':'custom','mathtext.rm':font,'mathtext.it':font+':italic','mathtext.bf':font+':bold','mathtext.fallback':None})
    def input(self,name):return self.root/saved_relative(name)
    def resolve(self,reference):
        if reference.startswith('saved/'):return self.input(Path(reference).name)
        if reference.startswith('docs/'):return REPO/reference
        if reference.startswith('outputs/'):return self.root/reference
        if reference.startswith('data/'):return self.data/Path(reference).name
        return self.source/reference
    def relative(self,path):
        path=Path(path)
        for prefix,root in [('workspace',self.root),('data',self.data),('repository',REPO)]:
            try:return prefix+'/'+str(path.relative_to(root))
            except ValueError:pass
        return 'external/'+path.name
    def table(self,name):return pd.read_csv(self.source/'tables'/f'{name}.csv')
    def figure(self,h,w=180):
        layout=self.layout_config.get(Path(self.current['output']).stem,{}) if self.current else {}
        padding=layout.get('bottom_padding_mm',0);h+=padding
        f=plt.figure(figsize=(w/25.4,h/25.4),dpi=150);f._padding_mm=padding;f._height_mm=h;f._width_mm=w;f._headings=[]
        f._panel_texts={};f._active_letter=None
        return f
    def ax(self,f,x,y,w,h,**kw):return f.add_axes([x/f._width_mm,(y+f._padding_mm)/f._height_mm,w/f._width_mm,h/f._height_mm],**kw)
    def text(self,f,x,y,text,**kw):
        return f.text(x/f._width_mm,(y+f._padding_mm)/f._height_mm,display_label(text),fontsize=kw.pop('fontsize',6.5),**kw)
    def heading(self,f,letter,title,x,y):
        # Titles live in the manuscript caption, not in the artwork.
        f._headings.append({'letter':letter,'description':display_label(title),'x_mm':x,'y_mm':y})
        if letter:
            layout=self.layout_config.get(Path(self.current['output']).stem,{}) if self.current else {}
            found=next((h for h in layout.get('headings',[]) if h.get('letter')==letter),{})
            if found.get('letter_position_mm'):
                xx,yy=found['letter_position_mm']
                f.text(xx/f._width_mm,yy/f._height_mm,letter,fontsize=8,fontweight='bold',va='baseline')
            else:self.text(f,x,y,letter,fontsize=8,fontweight='bold',va='top')
    def cb(self,f,artist,rect,label='',orientation='vertical',ticks=None):
        cb=f.colorbar(artist,cax=self.ax(f,*rect),orientation=orientation,ticks=ticks)
        if orientation=='vertical':cb.ax.set_title(label.replace('Signed SHAP','Signed\nSHAP').replace('Mean |pair SHAP|','Mean |pair\nSHAP|'),fontsize=6.5,pad=4)
        else:cb.set_label(label,fontsize=6.5,labelpad=3)
        cb.ax.tick_params(labelsize=6.5,length=1.5,pad=2)
        if cb.solids is not None:cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face')
        cb.outline.set_linewidth(.3)
        if orientation=='vertical':cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        return cb
    def panel(self,fig,letter,title,sources,detail,ci='None',transform='None',n=2164):
        paths=[self.resolve(x) for x in sources]
        self.panels.append({'figure':self.current['id'],'panel':letter,'description':display_label(title),'n_participants':n,
            'detail':display_label(detail),'ci':display_label(ci),'display_transform':display_label(transform),
            'sources':[self.relative(p) for p in paths],
            'source_sha256':{self.relative(p):digest(p) for p in paths}})
    def save(self,f,name):
        name=Path(self.current['output']).stem
        f.canvas.draw()
        for t in f.findobj(matplotlib.text.Text):
            if t.get_text():t.set_text(display_label(t.get_text()))
        for ax in f.axes:
            for which in ['x','y']:
                labels=[t.get_text() for t in getattr(ax,'get_'+which+'ticklabels')()]
                if any(display_label(t)!=t for t in labels):getattr(ax,'set_'+which+'ticks')(getattr(ax,'get_'+which+'ticks')(),[display_label(t) for t in labels])
            for t in [ax.title,ax.xaxis.label,ax.yaxis.label,*ax.texts]:t.set_text(display_label(t.get_text()))
        f.canvas.draw()
        for ax in f.axes:
            for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
                lo,hi=sorted(limits)
                for tick in axis.get_major_ticks():
                    if tick.get_loc()<lo-1e-9 or tick.get_loc()>hi+1e-9:tick.label1.set_visible(False);tick.label2.set_visible(False)
        f.canvas.draw();renderer=f.canvas.get_renderer();outside=[];sizes=[]
        for t in f.findobj(matplotlib.text.Text):
            if not t.get_visible() or not t.get_text():continue
            box=t.get_window_extent(renderer);sizes.append(t.get_fontsize())
            if box.x0<-.5 or box.y0<-.5 or box.x1>f.bbox.width+.5 or box.y1>f.bbox.height+.5:outside.append(t.get_text())
        if outside:raise ValueError(f'{name}: text exceeds canvas with selected font {self.font}: {outside}')
        axes=[]
        for i,ax in enumerate(f.axes):
            pos=ax.get_position();axes.append({'index':i,'x_mm':pos.x0*f._width_mm,'y_mm':pos.y0*f._height_mm,'width_mm':pos.width*f._width_mm,'height_mm':pos.height*f._height_mm,'xlabel':ax.get_xlabel(),'ylabel':ax.get_ylabel()})
        files=[]
        for suffix in ['pdf','png','svg']:
            p=self.figdir/f'{name}.{suffix}';f.savefig(p,dpi=300);files.append(p)
        plt.close(f);page=PdfReader(files[0]).pages[0]
        if list(page.images):raise ValueError(f'{name}: unexpected raster image in vector PDF')
        self.outputs.append({'id':self.current['id'],'width_mm':f._width_mm,'height_mm':f._height_mm,'font':self.font,
            'minimum_font_pt':min(sizes),'raster_images':0,'source_mode':'redrawn_from_saved_numeric_results',
            'files':{p.name:digest(p) for p in files}})
        self.layouts.append({'id':self.current['id'],'headings':f._headings,'axes':axes,'all_text_within_canvas':True})
        print(f'Rendered {self.current["id"]}: {name}',flush=True)
