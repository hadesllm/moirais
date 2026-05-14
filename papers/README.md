# morie papers

This directory contains the source LaTeX for the five papers that document the **morie** toolkit and its MRM (Multilevel Reconciliation Methodology) framework.

## Papers

| Paper | Source | Zenodo DOI |
|---|---|---|
| **morie (Python)** | [`morie-py-paper/main.tex`](morie-py-paper/main.tex) | [10.5281/zenodo.20096350](https://doi.org/10.5281/zenodo.20096350) |
| **morie (R)** | [`morie-r-paper/main.tex`](morie-r-paper/main.tex) | [10.5281/zenodo.20111233](https://doi.org/10.5281/zenodo.20111233) |
| **MRM framework** | [`mrm-formulations-paper/main.tex`](mrm-formulations-paper/main.tex) | [10.5281/zenodo.20096075](https://doi.org/10.5281/zenodo.20096075) |
| **MRM empirical** | [`morie-empirical-paper/main.tex`](morie-empirical-paper/main.tex) | [10.5281/zenodo.20175689](https://doi.org/10.5281/zenodo.20175689) |
| **Hawkes process** | [`hawkes-paper/main.tex`](hawkes-paper/main.tex) | [10.5281/zenodo.20102198](https://doi.org/10.5281/zenodo.20102198) |

All papers use the [JSS](https://www.jstatsoft.org/) document class via the bundled `jss.cls` + `jss.bst`. Compile with `xelatex main.tex` (or `pdflatex main.tex`) followed by `bibtex main && xelatex main.tex && xelatex main.tex`.

## What's tracked here vs. what isn't

This directory is **explicitly allowlisted** in the repo's `.gitignore`. Only the following files are tracked:

- `<paper>/main.tex` — paper source
- `<paper>/refs.bib` — paper bibliography
- `jss.cls`, `jss.bst` — JSS LaTeX class + bibstyle
- This `README.md`

Everything else (working drafts, frozen arxiv bundles, paper PDFs, results CSVs, email correspondence, anything under `azm/` or `dirty-frag-research/`) is the author's private working tree and stays out of the public repo. A previous public commit accidentally included email drafts, which had to be reverted at significant cost; the allowlist is the safety belt against that recurring.

## Version stamp

All five papers carry a `morie version stamp` comment block (right after `\Plainkeywords{}`) declaring the morie public-API version the paper documents. The current stamp is **morie v0.7.0**. The papers will be re-stamped on each major API revision.

## Citing

If you use morie or its findings, cite both software papers (R and Python) plus the framework paper. Where applicable, also cite the Hawkes methodology paper and the empirical applications paper. See the main repo's [`CITATION.cff`](https://github.com/hadesllm/morie/blob/main/CITATION.cff) for machine-readable metadata covering all five DOIs.
