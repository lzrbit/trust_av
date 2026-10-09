"""Generate S16--S20 from saved configuration and omnibus tables; no estimates."""
from pathlib import Path
import argparse, hashlib, json
import pandas as pd


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def esc(s):
    return str(s).replace('\\',r'\textbackslash{}').replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#')
def number(v):
    try: return f'{float(v):g}'
    except (ValueError,TypeError): return esc(v)
def pformat(v): return r'$<0.001$' if v<.001 else f'{v:.3f}'

def build_tables(results_dir, output_dir=None):
    dest=Path(output_dir or results_dir); tab=dest/'tables'; source=Path(results_dir)/'tables'; config=pd.read_csv(source/'parameter_configurations.csv'); omnibus=pd.read_csv(source/'parameter_omnibus_tests.csv'); outputs=[]
    for n,scenario in enumerate(['LC','SVM','HB','MB'],16):
        d=config[config.display_scenario.eq(scenario)]
        columns=['Lateral category','ACC category','Neighbour merging distance (m)'] if scenario=='LC' else ['Speed (km/h)',r'Braking (m/s$^2$)', {'SVM':'Ego merging distance (m)','HB':'Initial braking distance (m)','MB':'Merging distance (m)'}[scenario]]
        name=f'table_s{n}_configuration_{scenario.lower()}.tex'; label=f'tab:s{n}_configuration_{scenario.lower()}'
        lines=[r'% Generated from original assigned configurations; raw source fields retained in CSV.',r'\begin{table}[!htbp]',r'\centering\small',r'\setlength{\tabcolsep}{3pt}',r'\renewcommand{\arraystretch}{1.06}',r'\caption{'+scenario+' experimental configurations. Each row is one event design from the original workbook; scenario names follow those used in the manuscript.}',r'\label{'+label+'}',r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccc@{}}',r'\toprule','Source event & '+' & '.join(columns)+r' \\',r'\midrule']
        for r in d.itertuples(): lines.append(esc(r.source_event_name)+' & '+' & '.join(number(v) for v in [r.factor1_value,r.factor2_value,r.factor3_value])+r' \\')
        raw_headers=[d.iloc[0][f'factor{i}_source_header'] for i in [1,2,3]]
        note='Source: \\texttt{parameter\\_list.xlsx}, sheet \\texttt{'+esc(d.source_sheet.iloc[0])+'}. Exact original column headings, units and historical editing notes are retained in the source-cell export. The editing notes are not used to filter current observations.'
        if scenario=='LC': note+=' The original category \\texttt{Abortion} is displayed as Aborted in figures. Assigned distances for MAL\\_19--MAL\\_24 have a documented mismatch with detected manoeuvre-onset distances; the pre-existing exclusion sensitivity remains separate.'
        if scenario=='HB': note+=' HB\\_7--HB\\_9 have inconsistent stored acceleration-channel labels; these rows use the assigned braking intensity, which agrees with the velocity-derived acceleration check.'
        lines += [r'\bottomrule',r'\end{tabular*}',r'\par\smallskip',r'\begin{minipage}{\linewidth}\footnotesize\raggedright',note,r'\end{minipage}',r'\end{table}']
        (tab/name).write_text('\n'.join(lines)+'\n'); outputs.append({'table_number':n,'file':name,'label':label,'rows':len(d),'sha256':sha(tab/name)})
    names={'lateral_behaviour':'Lateral behaviour','acc_style':'ACC style','design_speed':'Speed','design_braking':'Braking intensity','design_distance':'Distance'}
    name='table_s20_parameter_omnibus.tex'; label='tab:s20_parameter_omnibus'
    lines=[r'% Newly estimated exploratory omnibus tests; not copied from prior ANOVA.',r'\begin{table}[!htbp]',r'\centering\small',r'\setlength{\tabcolsep}{2.5pt}',r'\renewcommand{\arraystretch}{1.12}',r'\caption{Exploratory overall associations of the experimental factors with perceived risk. Each test jointly examines all non-reference level coefficients across the scenario\textquotesingle s clips, adjusting for its other assigned factors and participant fixed intercepts.}',r'\label{'+label+'}',r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrrrr@{}}',r'\toprule',r'Scenario & Factor & Wald $\chi^2$ & df & $p$ & Holm $p$ & Partial $R^2$ \\',r'\midrule']
    for r in omnibus.itertuples(): lines.append(f'{r.display_scenario} & {names[r.factor]} & {r.wald_chi2:,.2f} & {r.df} & {pformat(r.p_raw)} & {pformat(r.p_holm_12)} & {r.weighted_within_partial_r2:.3f}'+r' \\')
    note=r'Each clip model gives each participant total weight one across their available positive event ratings. Joint covariance clusters on the same 2,164 participants across clips, with correction $G/(G-1)$; it retains cross-clip covariance. There are 51,584, 42,740, 42,891 and 42,751 valid event--clip ratings in LC, SVM, HB and MB, respectively. Degrees of freedom count all tested level-by-clip coefficients. Holm adjustment covers these 12 tests. Partial $R^2=(\mathrm{SSE}_{\rm reduced}-\mathrm{SSE}_{\rm full})/\mathrm{SSE}_{\rm reduced}$ uses the summed weighted within-participant residual sums of squares across clips; the reduced models omit only the tested factor. It measures conditional in-sample fit, not predictive performance. All estimates condition on the present stimulus library. The 195 paired comparisons used for boxplot annotations and the earlier 75-contrast bootstrap family are distinct analyses. Small $p$ values are displayed as $<0.001$, including values below floating-point resolution.'
    lines += [r'\bottomrule',r'\end{tabular*}',r'\par\smallskip',r'\begin{minipage}{\linewidth}\footnotesize\raggedright',note,r'\end{minipage}',r'\end{table}']; (tab/name).write_text('\n'.join(lines)+'\n'); outputs.append({'table_number':20,'file':name,'label':label,'rows':len(omnibus),'sha256':sha(tab/name)})
    report={'status':'pass','script_sha256':sha(Path(__file__)),'input_hashes':{p.name:sha(p) for p in [source/'parameter_configurations.csv',source/'parameter_omnibus_tests.csv']},'tables':outputs,'note':'Saved numeric values directly formatted; no statistical calculation or resampling.'}; (dest/'reports/parameter_table_manifest.json').write_text(json.dumps(report,indent=2)+'\n'); return report
