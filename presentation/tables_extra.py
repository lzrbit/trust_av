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


LABEL_FAMILY = {'mean_lat_z': 'Mean risk, LC', 'mean_long_z': 'Mean risk, SVM/HB/MB',
                'excursion_lat_z': 'Excursion, LC', 'excursion_long_z': 'Excursion, SVM/HB/MB',
                'late_lat_z': 'Late change, LC', 'late_long_z': 'Late change, SVM/HB/MB'}
CATEGORY_LABEL = {'acc_down_lks_up': 'ACC decreased, LKS increased', 'acc_up_lks_down': 'ACC increased, LKS decreased',
                  'both_up': 'Both increased', 'both_down': 'Both decreased',
                  'acc_zero_lks_nonzero': 'ACC unchanged, LKS changed', 'lks_zero_acc_nonzero': 'LKS unchanged, ACC changed',
                  'both_zero': 'Both unchanged'}


def _ci(low, high):
    return rf'$[{float(low):.3f},\ {float(high):.3f}]$'


def table21(direction, transitions):
    """Supplementary Table S21: joint direction of ACC/LKS changes and ordering transitions."""
    by = {r['category']: r for r in direction}
    n = int(by['acc_down_lks_up']['n_participants'])
    assert sum(int(by[k]['n']) for k in CATEGORY_LABEL) == n
    tr = {r['pre_relation']: r for r in transitions}
    order = ['acc_above', 'equal', 'lks_above']
    audit = []
    text = PREFIX + rf"""\begin{{table}}[!htbp]
\centering\small
\setlength{{\tabcolsep}}{{3.2pt}}
\renewcommand{{\arraystretch}}{{1.15}}
\caption{{Joint direction of within-person ACC and LKS trust-rating changes. Each participant is assigned to exactly one category by the signs of post-minus-pre ACC and LKS trust ratings ($n={n:,}$).}}
\label{{tab:s21_change_direction}}
\begin{{tabular*}}{{\linewidth}}{{@{{\extracolsep{{\fill}}}}lrrr@{{}}}}
\toprule
Category & $n$ & \% & 95\% CI (\%) \\
\midrule
"""
    for key, label in CATEGORY_LABEL.items():
        r = by[key]
        cells = [label, f"{int(r['n']):,}", f"{100*float(r['proportion']):.1f}",
                 rf"$[{100*float(r['ci_low']):.1f},\ {100*float(r['ci_high']):.1f}]$"]
        text += ' & '.join(cells) + r' \\' + '\n'
        audit.append({'category': key, 'source_values': r, 'display_cells': cells})
    text += r"""\midrule
\multicolumn{4}{@{}l}{Relation between LKS and ACC ratings before (rows) and after (columns) the task} \\
 & ACC $>$ LKS & Equal & LKS $>$ ACC \\
"""
    for key, label in [('acc_above', 'ACC $>$ LKS before'), ('equal', 'Equal before'), ('lks_above', 'LKS $>$ ACC before')]:
        cells = [label] + [f"{int(tr[key][c]):,}" for c in order]
        text += ' & '.join(cells) + r' \\' + '\n'
        audit.append({'pre_relation': key, 'source_values': tr[key], 'display_cells': cells})
    text += rf"""\bottomrule
\end{{tabular*}}
\par\smallskip
\begin{{minipage}}{{\linewidth}}\small\raggedright
Percentages are of all {n:,} participants; intervals are percentile limits from 2,000 participant-bootstrap draws. The categories are descriptive classifications of integer rating changes and do not test any difference. The lower block counts participants by the ordering of their two function-specific ratings before and after the task; equal ratings include zero on both items. These counts show that opposite-direction changes occur in a minority of participants and that the mean divergence reflects an imbalance between the two opposite-direction categories together with a net shift in ordering, not a uniform individual response.
\end{{minipage}}
\end{{table}}
"""
    return text, audit


def table22(coefficients, equality, post_coefficients, cross, pooled_refit):
    """Supplementary Table S22: scenario-family decomposition of the risk--trust associations."""
    c1 = {r['term']: r for r in coefficients}
    eq = {r['contrast']: r for r in equality}
    acc = {r['term']: r for r in post_coefficients if r['outcome'] == 'trust_post_acc'}
    lks = {r['term']: r for r in post_coefficients if r['outcome'] == 'trust_post_lks'}
    xc = {r['term']: r for r in cross}
    pooled = {r['term']: r for r in pooled_refit}['late_change_z']
    n = int(c1['late_long_z']['n'])
    audit = []
    text = PREFIX + rf"""\begin{{table}}[!htbp]
\centering\small
\setlength{{\tabcolsep}}{{3.0pt}}
\renewcommand{{\arraystretch}}{{1.15}}
\caption{{Scenario-family decomposition of the risk--trust associations. Risk summaries are computed separately within the lateral scenario (LC) and the three longitudinal scenarios (SVM, HB and MB), using the same cross-fitted high-point positions as the pooled summaries ($n={n:,}$ complete cases).}}
\label{{tab:s22_scenario_family_trust}}
\begin{{tabular*}}{{\linewidth}}{{@{{\extracolsep{{\fill}}}}lrrrr@{{}}}}
\toprule
\multicolumn{{5}}{{@{{}}l}}{{\textbf{{a}}, Within-person change contrast $D=(\mathrm{{LKS}}_{{\rm post}}-\mathrm{{LKS}}_{{\rm pre}})-(\mathrm{{ACC}}_{{\rm post}}-\mathrm{{ACC}}_{{\rm pre}})$}} \\
Risk predictor (per SD) & Coefficient & HC1 95\% CI & $p$ & BH $q$ \\
\midrule
"""
    for t, label in LABEL_FAMILY.items():
        r = c1[t]
        cells = [label, f"{float(r['estimate']):.3f}", _ci(r['ci_low'], r['ci_high']), probability(r['p_two_sided_normal']), probability(r['q_BH_family6'])]
        text += ' & '.join(cells) + r' \\' + '\n'
        audit.append({'panel': 'a', 'term': t, 'source_values': r, 'display_cells': cells})
    text += r"""\midrule
\multicolumn{5}{@{}l}{\textbf{b}, Equality of the LC and SVM/HB/MB coefficients in the same model} \\
Contrast & Difference & HC1 95\% CI & $p$ & BH $q$ \\
\midrule
"""
    for a, label in [('mean', 'Mean risk'), ('excursion', 'Excursion'), ('late', 'Late change')]:
        r = eq[f'{a}_lat_z - {a}_long_z']
        cells = [f'{label}: LC minus SVM/HB/MB', f"{float(r['estimate']):.3f}", _ci(r['ci_low'], r['ci_high']), probability(r['p_two_sided_normal']), probability(r['q_BH_3'])]
        text += ' & '.join(cells) + r' \\' + '\n'
        audit.append({'panel': 'b', 'contrast': r['contrast'], 'source_values': r, 'display_cells': cells})
    text += r"""\bottomrule
\end{tabular*}
\par\smallskip
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}
\toprule
\multicolumn{5}{@{}l}{\textbf{c}, Post-task ACC and LKS trust, each adjusted for its own initial rating} \\
Risk predictor (per SD) & ACC coefficient [95\% CI] & LKS coefficient [95\% CI] & LKS minus ACC [95\% CI] & BH $q$ \\
\midrule
"""
    for t, label in LABEL_FAMILY.items():
        a, l, x = acc[t], lks[t], xc[t]
        cells = [label, f"{float(a['estimate']):.3f} " + _ci(a['ci_low'], a['ci_high']), f"{float(l['estimate']):.3f} " + _ci(l['ci_low'], l['ci_high']),
                 f"{float(x['lks_minus_acc_estimate']):.3f} " + _ci(x['ci_low'], x['ci_high']), probability(x['q_BH_6'])]
        text += ' & '.join(cells) + r' \\' + '\n'
        audit.append({'panel': 'c', 'term': t, 'source_values': {'acc': a, 'lks': l, 'contrast': x}, 'display_cells': cells})
    text += rf"""\bottomrule
\end{{tabular*}}
\par\smallskip
\begin{{minipage}}{{\linewidth}}\small\raggedright
Panel \textbf{{a}} refits the Supplementary Table~\ref{{tab:s15_within_person_trust_contrast}} model with the three pooled risk summaries replaced by their scenario-family components; it retains response fraction, all six initial attitude items, age and experience per ten years, gender and country. Each family feature is standardised over all available participants before complete-case selection. Benjamini--Hochberg adjustment covers the six family coefficients in \textbf{{a}}, the three equality tests in \textbf{{b}}, and the six between-outcome contrasts in \textbf{{c}}, as separate families. Panel \textbf{{c}} fits one ordinary least-squares model per outcome on the same participants, with HC1 covariance; the between-outcome contrast uses the joint HC1 covariance of the two models. The pooled model refitted on these {n:,} participants gives a late-change coefficient of ${float(pooled['estimate']):.3f}$ (${float(pooled['ci_low']):.3f}$ to ${float(pooled['ci_high']):.3f}$). Family features are correlated with each other and with the pooled summaries, so coefficients are conditional associations within one specification. This analysis was added after review of the main results to examine whether the pooled late-change association depends on the scenario family; it is exploratory, is not a mediation or exposure-effect analysis, and does not remove the pre/post change in item referent.
\end{{minipage}}
\end{{table}}
"""
    return text, audit
