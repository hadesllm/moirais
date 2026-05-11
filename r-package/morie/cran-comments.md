# cran-comments.md

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.
* The "possibly misspelled words" sub-note flags 11 terms in DESCRIPTION that
  are all correctly spelled domain vocabulary, not typos:
  - **Statistical-method acronyms**: `AIPW` (augmented inverse-probability
    weighting), `ATT` (average treatment effect on the treated), `ATC` (on
    the controls), `CATE` (conditional average treatment effect),
    `MRM` (the McNamara-Ruhela-Medina framework, named after its three
    contributors).
  - **Proper nouns**: `Hawkes` (Alan G. Hawkes, originator of the self-exciting
    point process), `Rosenbaum` (Paul Rosenbaum, sensitivity-analysis
    methodology), `Ruhela` (the package author's family name).
  - **Domain-standard terms**: `carceral`, `cryptographic`, `psychometrics`.

## Test environments

* local macOS 26.4 (aarch64-apple-darwin25.4.0), R 4.6.0 — `R CMD check --as-cran`
  passes with 0 errors / 0 warnings / 2 notes (one is the new-submission flag,
  one is the local HTML Tidy/V8 environmental note that does not occur on CRAN
  infrastructure)
* win-builder R-release (Windows) — uploaded; result email pending
* win-builder R-devel (Windows) — uploaded; result email pending

## Submission notes

* Suggests includes `DoubleML`, `mlr3`, `mlr3learners` for `estimate_irm()`,
  which is gated with `requireNamespace()` so the package functions normally
  without them.
* The `data-raw/` directory is excluded via `.Rbuildignore`.
* Companion to the Python `morie` package on PyPI; both share the same
  conceptual public API (~99 R exports, parallel Python public surface).

## Reverse dependencies

This is a new package — no reverse dependencies.
