# Luận văn LaTeX

Dự án LaTeX này được tạo từ nội dung mã nguồn, tài liệu và kết quả thực nghiệm trong kho `MPaGE`.

Biên dịch bằng XeLaTeX:

```bash
cd thesis
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

Các hình trong `figures/` được sao chép từ `figure/` và `mpage_bmab/experiments/results/images/`.

Trong VSCode, LaTeX Workshop nên dùng recipe `xelatex -> bibtex -> xelatex x2`.
Nếu VSCode báo `spawn latexmk ENOENT` hoặc `spawn xelatex ENOENT`, máy chưa cài TeX distribution hoặc VSCode chưa thấy đường dẫn `/Library/TeX/texbin`.
