#!/usr/bin/env python3
"""Render manuscript figures and SI tables from authorized, saved numeric outputs."""
import argparse,hashlib,json,os,re,sys
from pathlib import Path
REPO=Path(__file__).resolve().parent
MAP=REPO/'config/manuscript_map.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,default=REPO/'work')
    ap.add_argument('--data-dir',type=Path)
    ap.add_argument('--font',default='Arial',help='Use Arial for manuscript typography; font must be installed or supplied locally.')
    ap.add_argument('--font-dir',type=Path,help='Directory with licensed TTF fonts; fonts are not bundled.')
    sub=ap.add_subparsers(dest='action',required=True)
    sub.add_parser('list')
    p=sub.add_parser('figures');p.add_argument('ids',nargs='*',help='Fig1..Fig5, FigS1..; omit to render all.')
    sub.add_parser('panels',help='Export all 38 panels from verified current Arial render outputs.')
    sub.add_parser('tables',help='Regenerate the complete SI table set from saved results.')
    p=sub.add_parser('prepare');p.add_argument('--items',nargs='+',choices=['prepost','positions'])
    p=sub.add_parser('check');p.add_argument('--inputs',action='store_true');p.add_argument('--outputs',action='store_true');p.add_argument('--manuscript-dir',type=Path)
    a=ap.parse_args();workspace=a.workspace.expanduser().resolve();data=a.data_dir.expanduser().resolve() if a.data_dir else workspace/'inputs/derived'
    config=json.loads(MAP.read_text());items=config['figures_and_tables']
    if a.action=='list':
        for r in items:print(f"{r['id']:10s} {r['output']}\n  {' '.join(r['command'])}")
        return
    if a.action=='check':
        from presentation.validate import check
        report=check(config,workspace,data,a.inputs,a.outputs,a.manuscript_dir)
        dest=workspace/'outputs/presentation/reports';dest.mkdir(parents=True,exist_ok=True)
        (dest/'manuscript_map_validation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2));raise SystemExit(0 if report['status']=='pass' else 1)
    if a.action=='panels':
        from presentation.panels import export_rendered
        export_rendered(workspace);return
    if a.action=='tables':
        from presentation.tables import build
        build(workspace);return
    from presentation.figures import FigureBuilder
    b=FigureBuilder(workspace,data,font=a.font,font_dir=a.font_dir)
    if a.action=='prepare':
        from presentation.prepare import prepare
        prepare(b,a.items);return
    selected=[r for r in items if r['kind'] in ['main_figure','supplement_figure'] and (not a.ids or r['id'] in a.ids)]
    unknown=set(a.ids)-{r['id'] for r in selected}
    if unknown:ap.error('Unknown figure IDs: '+', '.join(sorted(unknown)))
    for row in selected:
        b.current=row
        getattr(b,row['method'])(*row.get('args',[]))
    record={'status':'rendered_requires_visual_review','map_sha256':sha(MAP),'producer_hashes':{str(p.relative_to(REPO)):sha(p) for p in sorted((REPO/'presentation').glob('*.py'))},'figure_count':len(b.outputs),'outputs':b.outputs,'panels':b.panels,'fonts':b.fonts,'input_mode':'Numeric sources; no pre-rendered figures','model_fitting_or_new_bootstrap':False}
    tag='all' if not a.ids else '_'.join(a.ids)
    (b.reportdir/f'figure_manifest_{tag}.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    (b.reportdir/f'figure_layout_{tag}.json').write_text(json.dumps(b.layouts,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
