# Presentation

Showcase deck for the domain-partition pipeline (`presentation.tex`),
self-contained: it only depends on the PNGs in `figures/`.

## Build

```bash
cd slides
pdflatex presentation.tex      # run twice for outline/nav/footer counters
pdflatex presentation.tex
```

Produces `presentation.pdf` (14 slides). Speaker notes are attached to each
frame via `\note{}` (render them with a notes-enabled build if needed).

## Refresh the figures

The images are copies of selected plots from `output/plots/`. Regenerate the
plots and re-copy after pipeline changes:

```bash
python domain_partition.py data/T1_9/T1_9_hub_raw.stl --method ta tb xiao --plots
cp output/plots/{step01_3d_surface_hub,step02_unwrapped_hub,\
step05_framefield_hub,step06_singularities_hub,\
step07_streamline_integration_hub,step07_streamline_integration_hub_labeled,\
step08_simplification_hub_ta,step08_simplification_hub_ta_labeled,\
step09_postprocessing_hub_ta,step09_postprocessing_hub_ta_labeled,\
step09_postprocessing_hub_tb,step09_postprocessing_hub_xiao,\
step10_tfi_hub_ta,step11_tiled_hub_ta}.png slides/figures/
```

## Theme

Uses the built-in `Boadilla` beamer theme so it compiles anywhere. For the
`metropolis` look, install it (e.g. `texlive-beamertheme-metropolis`) and
change the `\usetheme{Boadilla}` line to `\usetheme{metropolis}`.
