# Trust and perceived risk in automated driving

Scientific analysis and manuscript-production code for *Shared risk patterns and divergent trust in automated driving: A systematic analysis of 179K ratings across 29 countries*.

The repository contains the actual numeric rendering code for **all 5 main figures, every lettered main panel, 30 of the 31 supplementary figures and all 22 supplementary tables**, together with the analyses producing their inputs. Supplementary Fig. S1 is author-supplied schematic artwork (see [its provenance note](docs/figure_s1_provenance.md)). [The manuscript map](config/manuscript_map.json) links each ID to its input files, producer, command and output. Figures are redrawn from numbers; the rendering commands do not copy old PDFs.

Original questionnaires, participant-level derived tables, OOF/SHAP records, model bundles and individual boxplot fliers are **not included**. Exact full reproduction needs authorized inputs. Included aggregates support runnable examples and inspection of all 189 parameter boxes, 195 paired comparisons and 12 joint-factor tests, with individual fliers removed, and the complete scenario-family trust decomposition and change-direction counts behind Supplementary Tables S21–S22 (`data/aggregate/scenario_family/`). They cannot reconstruct every manuscript panel on their own.

No reuse license has yet been selected; see [LICENSE_STATUS.md](LICENSE_STATUS.md).

## Environment

The study runtime used Python 3.12. Install dependencies in a separate environment:

```sh
python -m venv .venv
# Activate .venv using your platform's usual command.
python -m pip install -r requirements.txt
python tests/smoke.py --with-model
```

For numeric-results-to-figure/table production without model fitting, `requirements-presentation.txt` is sufficient. For the included aggregate examples alone, use `requirements-preview.txt`.

LightGBM needs OpenMP. On macOS an existing Homebrew `libomp` installation is normally sufficient; Linux wheels may require the distribution's OpenMP runtime. Only load model bundles you created or trust. Paper layouts require locally installed, legitimately obtained **Arial**; no fonts are bundled. Pass `--font-dir` to locate licensed TTF files. The renderer checks missing fonts and text outside the canvas instead of silently changing typography.

## Quick start with included aggregates

```sh
python run.py aggregate-figures
python manuscript.py list
python manuscript.py check
```

The example command writes HB-distance boxes, pre/post summaries and prediction-performance plots under `work/outputs/public_aggregate_figures/`. The HB preview explicitly omits individual outlier points. These examples are not the full typeset figures.

## Full analysis and manuscript reproduction

Use one private workspace outside the checkout. Obtain permission for the exact source workbook or supply the documented derived tables. No participant data are downloaded automatically.

```sh
# Original study workbook, read-only; optional when derived inputs are available.
python run.py --workspace /path/to/private-workspace --questionnaire /path/to/questionnaire.xlsx build

# Existing derived data can be supplied with --data-dir on subsequent commands.
python run.py --workspace /path/to/private-workspace metadata describe risk-distribution dynamics country
python run.py --workspace /path/to/private-workspace risk-models trust-models
python run.py --workspace /path/to/private-workspace trust-contrasts multiplicity prepost parameters differential-trust scenario-family whatif parameter-inference

# Render from the resulting saved numeric outputs, without refitting.
python manuscript.py --workspace /path/to/private-workspace prepare
python manuscript.py --workspace /path/to/private-workspace check --inputs
python manuscript.py --workspace /path/to/private-workspace figures
python manuscript.py --workspace /path/to/private-workspace tables
python manuscript.py --workspace /path/to/private-workspace panels
python manuscript.py --workspace /path/to/private-workspace check --inputs --outputs
```

`run.py --list` lists analysis stages; `--dry-run` prints their commands. Heavy prediction and what-if stages run only when explicitly selected. `manuscript.py figures Fig1 FigS28` rebuilds selected compound figures. `tables` regenerates the complete SI table set. `panels` exports all 38 standalone panels from verified current Arial renders, preserving the frozen physical clipping rules. `Makefile` provides equivalent `map`, `check`, `prepare`, `figures`, `tables`, `panels`, `examples` and `smoke` targets.

See [reproduction instructions](docs/reproducing_the_manuscript.md), [readable figure/table map](docs/manuscript_map.md), [analysis-stage registry](config/analysis_stages.json), [data schema](docs/data_schema.md), [machine-readable fields](docs/derived_schema.json), [methods](docs/methods.md) and [source provenance](docs/source_provenance.json). Sample checks deliberately enforce this study's cohort and questionnaire coding; a different survey is not a drop-in input.

Local outputs may contain sensitive individual records. The source questionnaire remains unchanged. Risk 0 denotes no operation and is excluded only from video-risk observations; trust/acceptance 0 remains valid. All 29 reported countries remain included. Existing LC distance and HB acceleration-channel caveats are preserved.

## Verification and limits

Release checks actually render every mapped figure and regenerate every table from the authors' authorized saved outputs in an isolated output directory. They also compare the portable parameter analysis with saved numerical results. Synthetic checks cover participant-first aggregation, bootstrap behavior, disjoint evaluation groups and LightGBM attribution additivity. The full heavy model-training/bootstrap pipeline is not rerun just to package a release. Run manifests record source/output hashes, fonts, geometry and display definitions; final artwork still needs visual review.

The 75 original bootstrap contrasts, 195 paired t-test comparisons and 12 joint-factor tests are separate families. Country coverage does not establish representativeness, SHAP is model attribution, and fixed-model input perturbations are not intervention effects.

## Separate vector-panel export

```sh
python analysis/export_main_panels.py --input-dir /path/to/parent-pdfs --config config/main_panel_export_config.json --output-dir /path/to/panels
python tests/test_panel_export.py
```

Whitespace trimming requires Pillow and Poppler (`pdftoppm`). Temporary PNGs measure margins only; final output remains original vector content at 1:1 scale.

The unified `manuscript.py panels` command verifies the current renderer version, manuscript map, five parent hashes and page sizes, then records a run-specific configuration with unchanged crop rules. It has no arbitrary hash-bypass option.

This separate tool clips/composes panels from explicitly supplied parent PDFs and checks their hashes. It does not replace numeric rendering. Its manifest describes clipping/composition and physical dimensions; no new estimates or intervals are created. Parent PDFs are not included.
