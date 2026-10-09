# Manuscript production contract

`config/manuscript_map.json` is the figure/panel/table registry. It identifies every main panel, every supplementary figure and every supplementary table by manuscript ID, producer, input and output. `config/analysis_stages.json` identifies the analysis commands that create the required numeric files. Paths are relative to one of three explicit roots: the repository, the private workspace, or the authorized derived-data directory.

The current target is **5 main figures (37 lettered panels plus the Fig. 1a inset), 31 supplementary figures (30 rendered here; Fig. S1 is author-supplied artwork) and 22 supplementary tables**. Tables S21–S22 come from the `scenario-family` stage (SI Section S11). The workflow has two explanatory parts. The new four scenario figures contain three parameter panels each. The first 27 SI figures and first 15 SI tables retain their previous numbering.

## Two reproduction modes

1. **Included aggregates:** `python run.py aggregate-figures` runs without participant data. It draws three illustrative plots. The current HB example uses initial braking distance, with individual outliers omitted explicitly. The supplied parameter box summaries, paired comparisons, omnibus summaries and stimulus configurations can be inspected directly. They do not contain participant IDs or individual fliers.
2. **Full manuscript:** run the scientific stages with authorized inputs, or place the same saved outputs under the documented workspace paths. Then `manuscript.py prepare`, `figures` and `tables` produce every mapped artifact from numbers. No old figure PDF is required. Full SHAP displays need locally generated held-out attribution records; exact boxplots need locally generated flier values. Those files remain private.

The aggregate files cannot reconstruct an individual scatterplot or a held-out model. The code makes this boundary explicit instead of substituting a different figure.

## Private workspace layout

```text
private-workspace/
  inputs/derived/          participants.parquet, events.parquet,
                          windows.parquet, source_participants.parquet
  outputs/core/           core tables, OOF/SHAP records, local model bundles
  outputs/parameters/     original 75-contrast/126-coefficient bootstrap analysis
  outputs/parameter_inference/  189 box summaries, 195 paired tests,
                               12 joint-factor tests, 105 configurations
  outputs/prepost/        six paired 2,000-draw summaries
  outputs/whatif/         held-out fixed-model sensitivity outputs
  outputs/presentation/  newly drawn PDF/PNG/SVG files and LaTeX tables
```

`--data-dir` can override `inputs/derived`. The original workbook is optional when the derived tables and necessary saved outputs are already available. The four accounting tables are reproducible directly from derived data with `python run.py metadata`; `source_participants.parquet` is needed for the raw-export count but is not published.

## Commands

```sh
# Inspect the complete ID/input/output registry without accessing data.
python manuscript.py list
python manuscript.py check

# Create scientific outputs. Expensive modeling remains an explicit choice.
python run.py --workspace /path/to/private-workspace metadata describe risk-distribution dynamics country
python run.py --workspace /path/to/private-workspace risk-models trust-models
python run.py --workspace /path/to/private-workspace trust-contrasts multiplicity prepost parameters differential-trust scenario-family whatif parameter-inference

# Rebuild only descriptive SD summaries used in the artwork. Reuses saved CIs.
python manuscript.py --workspace /path/to/private-workspace prepare

# Check expected inputs, then render figures or the complete table set.
python manuscript.py --workspace /path/to/private-workspace check --inputs
python manuscript.py --workspace /path/to/private-workspace figures
python manuscript.py --workspace /path/to/private-workspace figures Fig1 Fig3 FigS28
python manuscript.py --workspace /path/to/private-workspace tables
python manuscript.py --workspace /path/to/private-workspace panels
python manuscript.py --workspace /path/to/private-workspace check --inputs --outputs

# Optional: compare the map with the author's current TeX sources.
python manuscript.py --workspace /path/to/private-workspace check --manuscript-dir /path/to/Manuscript
```

Each figure command builds the whole corresponding compound figure, including its shared legends and scales. Table formatting currently runs the whole set; each table's numeric dependencies are recorded individually. The `panels` command turns the verified current Arial parents into all 38 independent panels. It checks the current renderer/map hashes, parent hashes and page sizes before rebinding only the parent hashes in a derived run configuration; the frozen crop rules remain identical. Pillow and Poppler (`pdftoppm`) measure whitespace with temporary images; the final output is unscaled vector clipping/composition. The lower-level exporter also accepts explicitly supplied frozen parent PDFs with their original hash configuration. Neither route is a new statistical redraw.

## Fonts and output verification

The paper's measured layouts use **Arial**. Install a legitimately obtained Arial font locally or pass `--font-dir /path/to/licensed-ttf-files`. Fonts are not distributed. The renderer fails on missing fonts or text outside the canvas. `--font` permits another installed family for adaptation, but a different font can require layout changes and is not claimed to reproduce the paper's typography. The aggregate examples use Matplotlib defaults.

Every render writes source SHA-256 values, output hashes, font metadata, axes geometry and statistical display definitions under `outputs/presentation/reports`. The PDFs contain vector marks and text. These engineering checks do not replace visual review. Numerical stages retain their own seeds, participant weighting, sample checks and validation reports.

## Scope of the release checks

The release was tested by actually rebuilding all mapped figures/tables from the authors' saved, authorized outputs in an isolated output directory. The new parameter-inference implementation was run and compared with the saved tables. Small synthetic tests cover participant-first aggregation, bootstrap behavior, held-out splits and model-attribution additivity. The full heavy risk/trust model training and historical bootstrap stages were not rerun merely to package this release; their executable code and required inputs are supplied/documented.

The original 75 paired bootstrap contrasts, the 195 paired t-test annotation family and the 12 joint-factor Wald tests are separate analyses. `parameter-inference` preserves that distinction. Its default stimulus input is the included validated 105-event configuration export. `run.py --parameter-xlsx ... parameter-inference` additionally checks the original workbook cells. Neither route silently changes the six documented LC distance discrepancies or the stored HB acceleration-channel discrepancy.
