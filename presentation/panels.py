"""Export native-scale panels only from this renderer's verified current outputs."""
from pathlib import Path
import importlib.util,json
from pypdf import PdfReader
from .base import REPO,digest

def export_rendered(workspace):
    root=Path(workspace).resolve();presentation=root/'outputs/presentation';reports=presentation/'reports'
    manifest_path=reports/'figure_manifest_all.json'
    m=json.loads(manifest_path.read_text());map_path=REPO/'config/manuscript_map.json'
    if m['map_sha256']!=digest(map_path):raise ValueError('Render manifest uses an outdated manuscript map; render again')
    expected_producers={str(p.relative_to(REPO)):digest(p) for p in sorted((REPO/'presentation').glob('*.py'))}
    if m.get('producer_hashes')!=expected_producers:raise ValueError('Renderer changed after figure production; render again')
    base_path=REPO/'config/main_panel_export_config.json';cfg=json.loads(base_path.read_text());original_rules=json.dumps(cfg['panels'],sort_keys=True)
    layout=json.loads((REPO/'config/figure_layout.json').read_text());outputs={p['id']:p for p in m['outputs']};verified=[]
    for item in json.loads(map_path.read_text())['figures_and_tables']:
        if item['kind']!='main_figure':continue
        record=outputs[item['id']];source=root/item['output'];name=source.name
        if record['font']!='Arial':raise ValueError('Frozen physical panel rules require Arial')
        current_hash=digest(source)
        if record['files'].get(name)!=current_hash:raise ValueError('Rendered parent has changed: '+name)
        page=PdfReader(source).pages[0];dimensions=[float(page.mediabox.width)*25.4/72,float(page.mediabox.height)*25.4/72]
        expected=[layout[source.stem]['width_mm'],layout[source.stem]['height_mm']]
        if any(abs(a-b)>1e-7 for a,b in zip(dimensions,expected)):raise ValueError('Parent page dimensions differ from frozen rules: '+name)
        verified.append({'id':item['id'],'file':name,'sha256':current_hash,'dimensions_mm':dimensions})
        cfg['source_sha256'][name]=current_hash
    if len(verified)!=5:raise ValueError('Five verified main parents required')
    assert json.dumps(cfg['panels'],sort_keys=True)==original_rules
    provenance={'original_frozen_config_sha256':digest(base_path),'run_figure_manifest_sha256':digest(manifest_path),'manuscript_map_sha256':digest(map_path),'renderer_hashes':expected_producers,'verified_parents':verified,'crop_rules_unchanged':True,'scope':'Source hashes rebound only to this renderer current, hash-checked Arial outputs with exact parent page dimensions; no arbitrary hash-bypass option.'}
    cfg['derived_run_provenance']=provenance
    derived=reports/'panel_config_verified_rendered.json';derived.write_text(json.dumps(cfg,indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('trust_av_panel_export',REPO/'analysis/export_main_panels.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.export(presentation/'figures',derived,presentation/'panels')
    (reports/'panel_derivation.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return result
