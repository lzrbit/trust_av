# Trust and perceived risk in automated driving

Scientific analysis code for *Shared risk patterns and divergent trust in automated driving: A systematic analysis of 179K ratings across 29 countries*.

This repository contains the study's data-building, descriptive, inference, prediction, held-out SHAP, controlled-parameter and model-sensitivity calculations. It does **not** contain original questionnaires, participant-level derived data, prediction/SHAP records, fitted model files, or a complete manuscript/PDF bundle. The supplied aggregate tables support three small plotting examples and the 15-test HB annotation family. They are not sufficient to recreate every manuscript panel.

No reuse license has yet been selected; see [LICENSE_STATUS.md](LICENSE_STATUS.md). Please do not describe this as licensed open-source software until the authors add a license.

## Environment

The tested study runtime used Python 3.12. Install dependencies in a separate environment:

```sh
python -m venv .venv
# Activate .venv using your platform's usual command.
python -m pip install -r requirements.txt
python tests/smoke.py --with-model
```

LightGBM needs OpenMP. On macOS an existing Homebrew `libomp` installation is normally sufficient (`brew install libomp`); Linux wheels may require the distribution's OpenMP runtime. Do not copy someone else's model environment or load untrusted pickle/joblib files. No fonts or other third-party source are bundled. Plot previews use Matplotlib's installed font defaults.

## Public aggregate examples

```sh
python run.py aggregate-figures
```

This generates HB braking boxes, pre/post item summaries and prediction-performance plots under `work/outputs/public_aggregate_figures/`. The 15 HB boxes use saved participant-first quartiles, medians and observed 1.5-IQR whiskers. Individual outlier values are intentionally absent from the public tables, and the preview states that they are not displayed. These are data-reproduction examples, not pixel-identical reconstruction of the typeset article.

## Reproduction with authorized participant data

Participant data are not downloaded automatically. Obtain permission and supply the documented derived tables, or the study's original workbook using the exact questionnaire layout. Replacing the input with another survey does not automatically make the study-specific column map, sample assertions or measurement definitions valid.

```sh
# Option A: original study workbook; keep the workspace outside the Git checkout.
python run.py --workspace /path/to/private-workspace --questionnaire /path/to/questionnaire.xlsx build

# Option B: already derived study data; no original workbook is needed.
python run.py --workspace /path/to/private-workspace --data-dir /path/to/derived describe risk-distribution dynamics country
```

Inputs are configurable. `--workspace` controls outputs; `--data-dir` overrides the derived input directory; otherwise derived tables are read/written at `<workspace>/inputs/derived`. The workbook is read-only. Keep using the same `--workspace` and `--data-dir` for later stages. `python run.py --list` lists stages; `--dry-run` prints commands without running them. Local outputs can contain sensitive individual records and must not be uploaded without a separate disclosure review.

Run dependencies in this order as needed:

| Stage | Prerequisites | Scope |
|---|---|---|
| `build` | Original workbook | Reconstructs completed/licensed cohort and event/clip tables; original workbook unchanged |
| `describe risk-distribution dynamics country` | Derived participant/event/window tables; `describe` also uses source-participant flow table | Participant-first profiles, descriptive summaries, risk/trust associations and country standardization |
| `risk-models` | Derived participant/event/window tables | **Heavy:** four generalization protocols, matched comparators, training-only processing/tuning and held-out TreeSHAP |
| `trust-models` | Derived participant/window tables | **Heavy:** post-task trust prediction with training-fold risk features |
| `trust-contrasts multiplicity` | Dynamics and trust-model outputs | Cross-outcome coefficient contrasts and multiplicity sensitivity |
| `prepost` | Derived participants and dynamics trust-change table | Six paired item summaries with 2,000 participant bootstrap draws |
| `parameters` | Derived participants/windows and supplied 105-event design | Full design-factor profiles, paired contrasts, additive participant fixed-effects sensitivity |
| `hb-boxes hb-tests` | Derived windows; `hb-boxes` also needs `parameters` outputs | Descriptive boxes and post hoc 15-test paired-mean/Holm annotation family |
| `differential-trust` | Derived participants and dynamics participant risk-feature table | Exploratory differential trust-change OLS/HC1 association |
| `whatif` | Locally fitted risk-model bundles, OOF predictions and fold membership | Fixed held-out model sensitivity under local training-support rules; no refitting |
| `mixed` | Derived events | Optional POSIX-only crossed random-intercept diagnostic |

`risk-models`, `trust-models` and `whatif` are explicit choices; no heavyweight fitting runs by default. Scientific outputs remain in descriptive folders under `<workspace>/outputs/`. `parameters` and `hb-boxes` retain study-cohort assertions (2,164 participants, 179,966 operated ratings), deliberately preventing silent reuse with an incompatible cohort.

See [data schema](docs/data_schema.md), [machine-readable fields](docs/derived_schema.json), [methods and limitations](docs/methods.md), and [source provenance](docs/source_provenance.json). Metadata/hash checks tied to private historical audit files have been replaced by run-local source checks. Estimators, seeds, weighting, contrasts and test families are preserved; the public code does not claim that a new execution was preregistered.

## Verification scope

The preparation checks cover syntax/imports, synthetic bootstrap/profile invariants, disjoint held-out groups and synthetic LightGBM attribution additivity; HB test output is also compared with the study's saved table. Aggregate examples are actually run. The entire expensive modeling/bootstrap pipeline is **not** rerun as part of public-package preparation. Exact numerical reproduction additionally depends on the authorized study inputs, compatible dependency versions and the documented fixed seeds.

Current paper-specific vector layout and participant-level scatter/beeswarm panels are not reconstructible from the public aggregate files alone. Relevant analysis/plotting algorithms remain in the scientific modules; display exports requiring individual data need authorized local inputs. No raw individual table is disguised as an aggregate release.

## Standalone vector-panel export

A separate tool extracts the specified manuscript panels from **user-supplied**
parent PDFs. The parent PDFs are not included. This is native-scale vector
clipping/composition, with explicit foreign-text exclusions and shared-key
components; it is not a redraw from numerical data. The configuration checks the
exact five parent-file hashes, so an unrelated or edited PDF is rejected.

```sh
python analysis/export_main_panels.py --input-dir /path/to/parent-pdfs --config config/main_panel_export_config.json --output-dir /path/to/panels
python tests/test_panel_export.py
```

The configuration describes 37 panels plus the Fig. 1a inset (38 exports).
Export dimensions are recorded in millimetres with scale fixed at 1. This
engineering operation adds no estimates, stars, confidence intervals or models.
The synthetic test checks dimensions, vector content, explicit text exclusion,
source immutability and rejection of a mismatched parent hash. Each real export
still requires visual review; the exporter labels its manifest accordingly.
