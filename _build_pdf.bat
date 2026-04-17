@echo off
echo === pass 1 ===
pdflatex -interaction=nonstopmode paper_jaiio.tex > _build_p1.log 2>&1
echo === biber ===
biber paper_jaiio > _build_biber.log 2>&1
echo === pass 2 ===
pdflatex -interaction=nonstopmode paper_jaiio.tex > _build_p2.log 2>&1
echo === pass 3 ===
pdflatex -interaction=nonstopmode paper_jaiio.tex > _build_p3.log 2>&1
echo === DONE ===
