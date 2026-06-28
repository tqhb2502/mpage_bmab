# LaTeX Thesis

This directory contains the pre-template LaTeX thesis source generated from the
source code, documentation, figures, tables, and experimental evidence available
for this study. The official SOICT submission version is maintained in
`../SOICT_DATN_Research_ENG_Template/`.

Repository: <https://github.com/tqhb2502/mpage_bmab>

Compile with XeLaTeX:

```bash
cd thesis
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

Figures in `figures/` are the thesis-local copies used by `main.tex`.

In VSCode, LaTeX Workshop should use the recipe
`xelatex -> bibtex -> xelatex x2`. If VSCode reports
`spawn latexmk ENOENT` or `spawn xelatex ENOENT`, the TeX distribution is not
installed or VSCode cannot find `/Library/TeX/texbin`.
