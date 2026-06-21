# LaTeX Thesis

This LaTeX project contains the graduation thesis generated from the source
code, documentation, figures, tables, and experimental evidence available in
the `MPaGE` repository.

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
