#!/usr/bin/env python3
"""Export independent vector panels at 1:1 physical scale from supplied PDFs.

Needs pypdf; optional whitespace trimming also requires Pillow and Poppler.
The JSON config explicitly defines native-mm clipping regions,
foreign heading/text exclusions and optional shared-key components. This is
vector clipping/composition, not a statistical redraw. Output is never rasterized
or rescaled; optional temporary rendering only measures whitespace. Content is
clipped by PDF graphics operators (not merely
by setting a CropBox). Shared legends/country labels are copied at native scale.
Inputs must be supplied by the user; this script contains no participant data.
"""
from pathlib import Path
import argparse,copy,csv,hashlib,json,subprocess,tempfile
from pypdf import PdfReader,PdfWriter,PageObject,Transformation
from pypdf.generic import ContentStream,NameObject,RectangleObject
MM=72/25.4

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def filter_text(page,positions,prefixes):
    cs=ContentStream(page.get_contents(),page.pdf);ops=cs.operations;result=[];removed=[]
    for i,(a,op) in enumerate(ops):
        if op not in (b'Tj',b'TJ'):result.append((a,op));continue
        value=str(a[0]) if op==b'Tj' else ''.join(str(x) for x in a[0] if isinstance(x,str))
        j=i-1
        while j>=0 and ops[j][1] not in (b'cm',b'ET'):j-=1
        x=y=None
        if j>=0 and ops[j][1]==b'cm':x=float(ops[j][0][4])/MM;y=float(ops[j][0][5])/MM
        exclude=any(value.startswith(p) for p in prefixes)
        if x is not None:exclude |= any(abs(x-p[0])<.15 and abs(y-p[1])<.3 for p in positions)
        if exclude:removed.append({'text':value,'origin_mm':[x,y]})
        else:result.append((a,op))
    cs.operations=result;page[NameObject('/Contents')]=cs;return removed

def trim_vector_page(page):
    # Rendering locates white margins only. The retained content is vector and
    # is translated, never sampled or rescaled.
    from PIL import Image,ImageChops
    with tempfile.TemporaryDirectory(prefix='panel_margin_') as temp:
        folder=Path(temp);p=folder/'page.pdf';w=PdfWriter();w.add_page(page);w.write(p)
        subprocess.run(['pdftoppm','-singlefile','-r','150','-png',str(p),str(folder/'page')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        im=Image.open(folder/'page.png').convert('RGB');delta=ImageChops.difference(im,Image.new('RGB',im.size,'white')).convert('L');bb=delta.point(lambda v:255 if v>3 else 0).getbbox()
        if bb is None:raise ValueError('Exported panel is blank')
        pw,ph=float(page.mediabox.width),float(page.mediabox.height)
        x0=max(0,bb[0]/im.width*pw-MM);x1=min(pw,bb[2]/im.width*pw+MM)
        y0=max(0,(1-bb[3]/im.height)*ph-MM);y1=min(ph,(1-bb[1]/im.height)*ph+MM)
        clipped=copy.copy(page);clipped.cropbox=RectangleObject([x0,y0,x1,y1]);clipped.trimbox=clipped.cropbox
        trimmed=PageObject.create_blank_page(width=x1-x0,height=y1-y0)
        trimmed.merge_transformed_page(clipped,Transformation().translate(-x0,-y0),expand=False)
        return trimmed,{'retained_bounds_before_translation_mm':[x0/MM,y0/MM,x1/MM,y1/MM],'padding_mm':1.0,'scale':1.0,'method':'PNG used only for whitespace measurement; original vector content retained'}

def export(input_dir,config_path,output_dir):
    cfg=json.loads(config_path.read_text());output_dir.mkdir(parents=True,exist_ok=True);readers={};records=[]
    for panel in cfg['panels']:
        name=panel['source_pdf'];src=input_dir/name
        if name not in readers:readers[name]=PdfReader(src)
        original=readers[name].pages[0]
        if len(readers[name].pages)!=1 or list(original.images):raise ValueError(f'One vector page required: {src}')
        if cfg.get('source_sha256',{}).get(name) and sha(src)!=cfg['source_sha256'][name]:raise ValueError(f'Frozen parent hash mismatch: {name}')
        width,height=panel['canvas_mm'];dest=PageObject.create_blank_page(width=width*MM,height=height*MM);parts=[]
        for comp in panel['components']:
            page=copy.copy(original);removed=filter_text(page,comp.get('exclude_text_positions_mm',[]),comp.get('exclude_text_prefixes',[]))
            x0,y0,x1,y1=comp['crop_mm'];atx,aty=comp['at_mm'];assert x1>x0 and y1>y0
            page.cropbox=RectangleObject([x0*MM,y0*MM,x1*MM,y1*MM]);page.trimbox=page.cropbox
            dest.merge_transformed_page(page,Transformation().translate((atx-x0)*MM,(aty-y0)*MM),expand=False)
            parts.append({**comp,'removed_text_operations':removed,'scale':1.0})
        trim=None
        if cfg.get('trim_whitespace',False):dest,trim=trim_vector_page(dest)
        path=output_dir/panel['file'];path.parent.mkdir(parents=True,exist_ok=True);writer=PdfWriter();writer.add_page(dest);writer.pages[0].compress_content_streams();writer.add_metadata({'/Title':panel['id'],'/Subject':'Native-scale vector panel extraction; exact parent clipping and shared-key composition; no re-estimation'});writer.write(path)
        output=PdfReader(path).pages[0];assert not list(output.images)
        fonts=[str(f.get_object().get('/BaseFont')) for f in output['/Resources'].get('/Font',{}).values()]
        records.append({'id':panel['id'],'file':str(path.relative_to(output_dir)),'source_pdf':name,'source_sha256':sha(src),'sha256':sha(path),'bytes':path.stat().st_size,'width_mm':float(output.mediabox.width)/MM,'height_mm':float(output.mediabox.height)/MM,'native_scale':1.0,'raster_images':0,'fonts':sorted(set(fonts)),'method':'vector clipping with text-operation exclusions; optional 1:1 shared-component composition','components_before_whitespace_trim':parts,'whitespace_trim':trim,'notes':panel.get('notes','')})
        print(panel['id'],flush=True)
    manifest={'status':'pending_visual_QA','export_count':len(records),'config_sha256':sha(config_path),'script_sha256':sha(__file__),'native_scale_all':1.0,'no_rasterisation':True,'exports':records}
    (output_dir/'panel_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    keys=['id','file','width_mm','height_mm','native_scale','source_pdf','source_sha256','sha256','bytes']
    with (output_dir/'panel_dimensions_mm.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in records)
    return manifest
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();export(a.input_dir.resolve(),a.config.resolve(),a.output_dir.resolve())
