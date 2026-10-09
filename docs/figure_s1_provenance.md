# Supplementary Figure S1 provenance

Supplementary Fig. S1 (`figures/figure_s_workflow.pdf` in the manuscript project) is author-supplied vector artwork: a schematic of the experimental pipeline from the four scripted scenario types, through the IPG CarMaker event library and the division of each event video into six-second clips, to the driver-perspective rating interface with its 0–10 slider.

It is not rendered by this repository. It contains no data-derived numbers; every count shown in the manuscript (2,341 starters, 2,164 retained participants, 181,776 assigned clip ratings, 179,966 operated ratings, 34,624 assigned and 34,566 analysed participant–event sequences) is documented in the Supplementary Information text (Section S1.5) and in `outputs/core/tables/d2_sample_flow.csv`, produced by the `metadata` stage.

The earlier generated version of this figure (procedure and participant/response accounting, manuscript v8) was produced by `presentation.figures.FigureBuilder.workflow`; that renderer is retained for the accounting identity checks it performs but is no longer referenced by the manuscript map.
