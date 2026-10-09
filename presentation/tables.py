"""Regenerate LaTeX table fragments from saved scientific estimates."""
import json
from pathlib import Path
from . import tables_core,tables_extra

def build(workspace):
    root=Path(workspace).resolve()
    tables_core.main(root)
    tables=root/'outputs/presentation/tables';reports=root/'outputs/presentation/reports'
    manifest_path=reports/'supplement_table_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    for number,name,sourcefile,fn in [
        (14,'table_s14_parameter_contrasts.tex','controlled_parameter_paired_contrasts.csv',tables_extra.table14),
        (15,'table_s15_within_person_trust_contrast.tex','within_person_trust_contrast_risk3.csv',tables_extra.table15),
    ]:
        source=root/'outputs/parameters/tables'/sourcefile
        content,audit=fn(tables_extra.rows(source));target=tables/name;target.write_text(content)
        manifest['tables'].append({'number':number,'file':str(target.relative_to(root)),
            'sha256':tables_extra.sha(target),'sources':[{'path':str(source.relative_to(root)),'sha256':tables_extra.sha(source)}],
            'n_display_rows':len(audit),'conversion':'Saved values formatted; no new estimation'})
    family=root/'outputs/parameters/tables'
    for number,name,fn,sourcefiles in [
        (21,'table_s21_change_direction.tex',tables_extra.table21,['scenario_family_direction_classification.csv','scenario_family_rank_relation_transitions.csv']),
        (22,'table_s22_scenario_family_trust.tex',tables_extra.table22,['scenario_family_D_coefficients.csv','scenario_family_D_equality_tests.csv','scenario_family_post_trust_coefficients.csv','scenario_family_post_trust_cross_outcome_contrasts.csv','scenario_family_pooled_refit.csv']),
    ]:
        sources=[family/f for f in sourcefiles]
        content,audit=fn(*[tables_extra.rows(f) for f in sources]);target=tables/name;target.write_text(content)
        manifest['tables'].append({'number':number,'file':str(target.relative_to(root)),
            'sha256':tables_extra.sha(target),'sources':[{'path':str(f.relative_to(root)),'sha256':tables_extra.sha(f)} for f in sources],
            'n_display_rows':len(audit),'conversion':'Saved values formatted; no new estimation'})
    from .tables_parameters import build_tables
    extra=build_tables(root/'outputs/parameter_inference', root/'outputs/presentation')
    for r in extra['tables']:
        r['file']='outputs/presentation/tables/'+r['file']
        r['sources']=[{'path':'outputs/parameter_inference/tables/'+name,'sha256':value} for name,value in extra['input_hashes'].items()]
        manifest['tables'].append(r)
    manifest.update(status='pass',output_count=len(manifest['tables']),producer_hashes={p.name:tables_extra.sha(p) for p in sorted(Path(__file__).parent.glob('*tables*.py'))})
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
