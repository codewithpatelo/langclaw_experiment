# `guide/` — Authoring and Submission Material

This directory collects the material used to **author** the paper and
guide its revisions: style references, the author's canonical
theoretical definitions, conference-template files, and
bibliography / review PDFs consulted during drafting.

It is **not** part of the runtime code. If you came here to run the
experiment, see [`../README.md`](../README.md) and
[`../EXPERIMENT_SUMMARY.md`](../EXPERIMENT_SUMMARY.md) instead.

---

## 1. Authoring guides (read these first)

| file                          | purpose                                                                                       |
|-------------------------------|-----------------------------------------------------------------------------------------------|
| `writing_style.md`            | Author's scientific writing voice, distilled from Driveplexity iterations. Reference for AI assistants and co-authors. |
| `paper_benchmarks.md`         | Self-evaluation rubric for submission readiness (craft quality + lasting-impact dimensions).  |
| `TAH_axiomas.md`              | **Canonical, author-authoritative definition** of the Axioms of Homeostatic Autonomy (A1–A3 and Pr1). Do not edit based on reviewer paraphrases; update only if the author restates them. |
| `scientific_theory.md`        | Background notes and reasoning for the theoretical framing adopted in the paper.              |
| `guia.txt`                    | JAIIO 2026 formatting and length rules (short / long article variants).                       |

---

## 2. Conference templates (LNCS)

Files distributed by Springer for the Lecture Notes in Computer
Science class, vendored here so the paper compiles out of the box.

| file                     | purpose                                      |
|--------------------------|----------------------------------------------|
| `llncs.cls`              | LaTeX2e document class used by both papers.  |
| `splncs04.bst`           | LNCS BibTeX style (alphabetic sorting).      |
| `samplepaper.tex`        | Reference LNCS layout with standard sections.|
| `fig1.eps`               | Figure used by `samplepaper.tex`.            |
| `thebiblio.bib`          | Bibliography used by `samplepaper.tex`.      |
| `history.txt`            | Version history of the LNCS package.         |
| `readme.txt`             | Upstream Springer LNCS README (unmodified).  |
| `llncsdoc.pdf`           | Full LNCS class documentation.               |
| `ORCID-iD_icon_16x16.png`| ORCID icon for the author block.             |

---

## 3. Reference and review PDFs

Background reading and prior-round reviews that informed the current
draft. These are consulted material, not output of this repo.

| file                                                   | purpose                                                                 |
|--------------------------------------------------------|-------------------------------------------------------------------------|
| `paper_review.pdf`                                     | Aggregated reviewer comments from AI Reviewer 3 and peer feedback.      |
| `DEEPRESEARCH3.pdf`                                    | Deep-research brief consulted during framing.                           |
| `2-ASAID_HermannVignoloGerard.pdf`                     | Example ASAID-track short paper (structure and tone reference).         |
| `Markopoulos_17-Modeling decision-making -Gerpe_Patricio_2086.pdf` | Prior work by the author on decision-making modelling (context).        |
| `metrics.pdf`                                          | Reference on argumentation / MAS evaluation metrics.                    |
| `ref_goo.pdf`                                          | Supporting reference consulted while drafting.                          |
| `TP2_CyT (1).pdf`                                      | Prior course-level report used as context.                              |

---

## 4. How to use this folder when editing the paper

1. Before rewriting theory sections: re-read `TAH_axiomas.md`. The
   axioms are **ontological commitments**, not reviewer-driven
   paraphrases of the experimental mechanism.
2. When checking tone and cohesion: use `writing_style.md` as the
   reference voice and `paper_benchmarks.md` as the scoring rubric.
3. When checking format compliance: use `guia.txt` (JAIIO rules) and
   `samplepaper.tex` (LNCS layout).
4. When addressing reviewer comments: cross-check `paper_review.pdf`
   to avoid re-introducing previously addressed issues.
