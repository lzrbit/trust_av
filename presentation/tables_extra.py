"""Additional parameter and differential-trust tables from saved estimates."""

import csv,hashlib,json

from pathlib import Path

PREFIX = '% Generated from saved CSV estimates by presentation/tables_extra.py.\n'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def probability(value):
    value = float(value)
    if value >= .001:
        return f'{value:.3f}'
    mantissa, exponent = f'{value:.2e}'.split('e')
    return rf'${mantissa}\times 10^{{{int(exponent)}}}$'

def table14(data):
    data = [x for x in data if x['analysis_variant'] == 'assigned_design_primary']
    assert len(data) == 75
    groups = {}
    for row in data:
        key = (row['display_scenario'], row['factor'], row['reference_level'], row['comparison_level'])
        groups.setdefault(key, []).append(row)
    assert len(groups) == 14
    factor_names = {'design_speed': 'Cruise speed', 'design_braking': 'Braking',
                    'design_distance': 'Distance', 'lateral_behaviour': 'Lateral behaviour',
                    'acc_style': 'ACC category'}
    text = PREFIX + r'''\begin{table}[!htbp]
\centering\small
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.15}
\caption{Within-participant marginal risk differences by scenario, design factor and clip. Each entry is the mean difference in operated risk ratings for the comparison level minus the reference level, using the same participants at both levels at that clip. Other design factors are marginalised rather than matched.}
\label{tab:s14_parameter_contrasts}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}p{.065\linewidth}p{.135\linewidth}p{.23\linewidth}rrrrrr@{}}
\toprule
Scenario & Factor & Comparison vs reference & Clip 1 & Clip 2 & Clip 3 & Clip 4 & Clip 5 & Clip 6 \\
\midrule
'''
    previous = None
    audit = []
    for (scenario, factor, reference, comparison), group in groups.items():
        if previous and scenario != previous:
            text += r'\addlinespace[3pt]' + '\n'
        previous = scenario
        clips = {int(row['clip']): row for row in group}
        expected = 6 if scenario == 'LC' else 5
        assert sorted(clips) == list(range(1, expected + 1))
        if factor == 'lateral_behaviour':
            labels = {'1m/s': r'normal (1 m s$^{-1}$)', '3m/s': r'normal (3 m s$^{-1}$)',
                      'Fragmented': 'fragmented', 'Abortion': 'aborted'}
            comparison_text = labels[comparison].capitalize() + ' vs ' + labels[reference]
        elif factor == 'acc_style':
            comparison_text = comparison.capitalize() + ' vs ' + reference
        else:
            unit = {'design_speed': r'km h$^{-1}$', 'design_braking': r'm s$^{-2}$', 'design_distance': 'm'}[factor]
            comparison_text = f'{float(comparison):g} vs {float(reference):g} ' + unit
        values = [f'{float(clips[k]["mean_difference"]):.3f}' if k in clips else r'\textemdash' for k in range(1, 7)]
        text += ' & '.join([scenario, factor_names[factor], comparison_text] + values) + r' \\' + '\n'
        audit.append({'scenario': scenario, 'factor': factor, 'reference': reference, 'comparison': comparison,
                      'clips': sorted(clips), 'display_values': values,
                      'source_row_keys': [clips[k]['row_key'] for k in sorted(clips)]})
    text += r'''\bottomrule
\end{tabular*}
\par\smallskip
\begin{minipage}{\linewidth}\small\raggedright
Differences are in rating points (operated scores 1--10); no-operation zeros are omitted. The dash denotes an absent sixth clip, not a missing estimate. Pointwise intervals, the simultaneous sensitivity across all 75 primary contrasts, and clip-specific paired participant counts are provided in Supplementary Data. The design distance means neighbour-vehicle merging distance in LC, ego-vehicle merging distance in SVM, initial braking distance in HB, and merging distance in MB. LC distance uses assigned design values; a separate sensitivity excludes the six disputed events. These marginal comparisons do not isolate a factor while holding every other factor fixed. Table~\ref{tab:s2_contrasts} instead compares clip positions, and Table~\ref{tab:s7_country_profiles} reports country-level summaries.
\end{minipage}
\end{table}
'''
    return text, audit

def table15(data):
    assert len(data) == 3 and {x['term'] for x in data} == {'mean_risk_z', 'excursion_z', 'late_change_z'}
    byterm = {x['term']: x for x in data}
    text = PREFIX + r'''\begin{table}[!htbp]
\centering\small
\setlength{\tabcolsep}{3.2pt}
\renewcommand{\arraystretch}{1.15}
\caption{Post-review exploratory associations with the within-person difference in trust-rating changes. The outcome is $D=(\mathrm{LKS}_{\rm post}-\mathrm{LKS}_{\rm pre})-(\mathrm{ACC}_{\rm post}-\mathrm{ACC}_{\rm pre})$. Coefficients are rating points per sample standard deviation of the risk predictor.}
\label{tab:s15_within_person_trust_contrast}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrr@{}}
\toprule
Risk predictor & Coefficient & HC1 95\% CI & $p$ & BH $q$ & $N$ \\
\midrule
'''
    audit = []
    for term, label in [('mean_risk_z', 'Mean risk'), ('excursion_z', 'Excursion'), ('late_change_z', 'Late change')]:
        row = byterm[term]
        cells = [label, f'{float(row["estimate"]):.3f}',
                 rf'$[{float(row["ci_low"]):.3f},\ {float(row["ci_high"]):.3f}]$',
                 probability(row['p_two_sided_normal']), probability(row['q_BH_risk3']), f'{int(row["n"]):,}']
        text += ' & '.join(cells) + r' \\' + '\n'
        audit.append({'term': term, 'source_values': row, 'display_cells': cells})
    text += r'''\bottomrule
\end{tabular*}
\par\smallskip
\begin{minipage}{\linewidth}\small\raggedright
One ordinary least-squares model uses 2,158 complete cases and adjusts jointly for the three risk summaries, response fraction, all six initial attitude items (overall, ACC and LKS trust; delegation; raw supervision need; and non-driving activities), age and licence-based experience per ten years, gender and country. Risk summaries reuse the saved five-fold cross-fitted high-point definitions; no peak is reselected. Intervals and two-sided tests use HC1 covariance and the normal reference distribution. Benjamini--Hochberg adjustment applies only to these three risk coefficients. The full 43 coefficients, including the intercept and adjustment terms, are supplied in Supplementary Data. This analysis was specified after manuscript review, not preregistered; it describes an association and does not identify a cause of opposing mean changes or remove the pre/post change in item referent.
\end{minipage}
\end{table}
'''
    return text, audit
