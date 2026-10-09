"""Integrity checks for the declared manuscript-production contract."""
import ast,hashlib,re,json
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]

def check(config,workspace,data,check_inputs=False,check_outputs=False,manuscript_dir=None):
    rows=config['figures_and_tables'];errors=[];warnings=[]
    ids=[r['id'] for r in rows];outs=[r['output'] for r in rows]
    if len(ids)!=len(set(ids)):errors.append('Duplicate manuscript ID')
    if len(outs)!=len(set(outs)):errors.append('Duplicate output path')
    module=ast.parse((REPO/'presentation/figures.py').read_text())
    methods={n.name for n in ast.walk(module) if isinstance(n,ast.FunctionDef)}
    dependencies={}
    for r in rows:
        if r['kind'].endswith('figure'):
            if r['method'] not in methods:errors.append(r['id']+': missing renderer '+r['method'])
            panels=r['panels'];letters=[p['letter'] for p in panels]
            if len(letters)!=len(set(letters)):errors.append(r['id']+': duplicate panel letter')
            if r['kind']=='main_figure':
                expected={'Fig1':list('abcdefgh')+['a-inset'],'Fig2':list('abcdefgh'),'Fig3':list('abcdefgh'),'Fig4':list('abcdef'),'Fig5':list('abcdefg')}[r['id']]
                if set(letters)!=set(expected):errors.append(r['id']+': incomplete panel coverage')
            inputs=[q for p in panels for q in p['inputs']]
        else:inputs=r['inputs']
        if not inputs:errors.append(r['id']+': no declared numeric/protocol input')
        for q in inputs:
            if q['base'] not in ['workspace','data','repository']:errors.append(r['id']+': unknown input root');continue
            p={'workspace':workspace,'data':data,'repository':REPO}[q['base']]/q['path']
            dependencies[str(p)]={'relative':q['base']+'/'+q['path'],'producer':q.get('producer')}
            if check_inputs and not p.is_file() and q.get('producer')!='generated_during_render':errors.append(r['id']+': missing '+q['base']+'/'+q['path'])
        if check_outputs and not (workspace/r['output']).is_file():errors.append(r['id']+': missing output '+r['output'])
        if Path(r['output']).is_absolute() or '..' in Path(r['output']).parts:errors.append(r['id']+': unsafe output')
    stage_config=json.loads((REPO/config['analysis_config']).read_text())
    stage_ids={r['id'] for r in stage_config['stages']}
    for r in stage_config['stages']:
        if not (REPO/r['script']).is_file():errors.append('Missing analysis script: '+r['script'])
        if not set(r['requires_stages'])<=stage_ids:errors.append('Unknown prerequisite in '+r['id'])
    if check_outputs:
        reportdir=workspace/'outputs/presentation/reports'
        fp=reportdir/'figure_manifest_all.json'
        if not fp.exists():errors.append('Full figure manifest missing; render all figures for a complete output audit')
        else:
            fm=json.loads(fp.read_text())
            current_map=hashlib.sha256((REPO/'config/manuscript_map.json').read_bytes()).hexdigest()
            if fm.get('map_sha256')!=current_map:errors.append('Figure manifest uses a stale manuscript map')
            for name,digest in fm.get('producer_hashes',{}).items():
                if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=digest:errors.append('Figure producer changed since rendering: '+name)
            for r in fm['outputs']:
                for name,digest in r['files'].items():
                    p=workspace/'outputs/presentation/figures'/name
                    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:errors.append('Figure output hash mismatch: '+name)
        tp=reportdir/'supplement_table_manifest.json'
        if not tp.exists():errors.append('Table manifest missing')
        else:
            tm=json.loads(tp.read_text())
            for name,digest in tm.get('producer_hashes',{}).items():
                if hashlib.sha256((REPO/'presentation'/name).read_bytes()).hexdigest()!=digest:errors.append('Table producer changed since formatting: '+name)
            for r in tm['tables']:
                p=workspace/r['file']
                if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:errors.append('Table output hash mismatch: '+r['file'])
    if manuscript_dir:
        main=(manuscript_dir/'main.tex').read_text();supp=(manuscript_dir/'supplementary.tex').read_text()
        refs=lambda text:set(re.findall(r'\\includegraphics(?:\[[^]]*\])?\{figures/([^}]+)\}',text))
        table_refs=set(re.findall(r'\\input\{tables/([^}]+)\}',supp));table_refs={p if p.endswith('.tex') else p+'.tex' for p in table_refs}
        for kind,actual in [('main_figure',refs(main)),('supplement_figure',refs(supp)),('supplement_table',table_refs)]:
            declared={Path(r['output']).name for r in rows if r['kind']==kind}
            if declared!=actual:errors.append(f'{kind} coverage differs: missing={sorted(actual-declared)} extra={sorted(declared-actual)}')
    return {'status':'pass' if not errors else 'fail','scope':'Declared renderer/output/panel coverage'+('; numeric-input presence' if check_inputs else '')+('; output presence' if check_outputs else ''),'counts':{k:sum(r['kind']==k for r in rows) for k in ['main_figure','supplement_figure','supplement_table']},'unique_input_files':len(dependencies),'errors':errors,'warnings':warnings,'model_refit':False}
