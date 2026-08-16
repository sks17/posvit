# Paper source

This folder is self-contained and Overleaf-ready. Drag the whole `paper` folder into a
blank Overleaf project, or upload its contents preserving the `figures/` subfolder, then
set `paper.tex` as the main document.

```
paper.tex               the manuscript
cvpr.sty                CVPR template, loaded as \usepackage[review]{cvpr}
ieee_fullname.bst       bibliography style
egbib.bib               bibliography
figures/                every figure the manuscript includes
paper-submitted.pdf     the compiled version, for reference
```

Everything else the preamble loads (`graphicx`, `amsmath`, `amssymb`, `booktabs`,
`hyperref`, `cleveref`) ships with Overleaf's TeX Live, so no further uploads are needed.

To build locally instead:

```bash
cd paper && pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

To switch from the review version to camera-ready, edit the preamble of `paper.tex`:
`\usepackage[review]{cvpr}` becomes `\usepackage{cvpr}`.

## Figures

Every figure except the method diagram is produced by the pipeline into
`results/derived/figures/`, so refreshing them is a copy rather than a redraw:

| Figure | Produced by |
|---|---|
| `02_forest_powered_null.pdf` | `posvit equivalence` then `posvit figures` |
| `03_dose_response.pdf` | `posvit dose` then `posvit figures` |
| `06_object_size.pdf` | `posvit stratify` then `posvit figures` |
| `12_waterbirds_generality.pdf` | `posvit waterbirds` then `posvit figures` |
| `09_specificity_curve.pdf` | **no generator yet** — see below |
| `pipeline.png` | hand-drawn method diagram, not generated |

```bash
posvit figures
cp results/derived/figures/*.pdf paper/figures/
```

`09_specificity_curve.pdf` is the one gap. `posvit controls` writes the numbers it needs
into `results/derived/metrics/specificity__<safe_id>.json`, including the
matched-accuracy and matched-confidence excess for every control, but `posvit/figures.py`
has no `figure_specificity` yet. The version bundled here came from the original analysis
scripts.
