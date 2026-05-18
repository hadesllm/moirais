## Submission

This is morie 0.9.4.

morie is a multi-domain toolkit for observational inference and
intervention analysis, hosting the MRM (Multilevel Reconciliation
Methodology) framework for Canadian carceral, police, and oversight
data as its primary application.

morie is not currently on CRAN. This submission brings the package to
CRAN in line with the releases already published on PyPI (the Python
companion), the `hadesllm` r-universe, and GHCR.

## Test environments

* local macOS 26 (Darwin 25.4.0), R 4.6.0
* win-builder R-release and R-devel
* GitHub Actions: ubuntu-latest (R-release, R-oldrel-1, R-devel),
  macos-latest (R-release), windows-latest (R-release)
* r-universe (Linux + macOS + Windows binaries) -- the `hadesllm`
  r-universe builds morie continuously; `R CMD check` reported OK on
  the linux-devel x86_64 and arm64 runners.

## R CMD check results

`R CMD check --as-cran` on the 0.9.4 source tarball (local macOS 26,
R 4.6.0):

Status: 0 ERROR, 0 WARNING, 1 NOTE.

The single NOTE is the standard CRAN-incoming feasibility check:

```
* checking CRAN incoming feasibility ... NOTE
Maintainer: 'Vansh Singh Ruhela <hadesllm@proton.me>'
New submission
```

This is expected -- morie is not currently on CRAN.

## Compiled code

morie 0.9.1 introduced a shared C++ computational backend
(`libmorie`) -- a header-only numeric core compiled into the package
through Rcpp (`LinkingTo: Rcpp`). It is standard, portable C++ with no
system dependencies beyond a C++ compiler, and it builds on the
r-universe Linux, macOS, and Windows runners. The same core also
backs the Python companion package, so the two language sides share
one numerical implementation.

## Reverse dependencies

There are no reverse dependencies on CRAN.

## Dependencies

There are no hard runtime dependencies beyond the base + recommended
set plus `Rcpp`. Every other package -- spatial, time-series, machine
learning, psychometric, and signal-processing tooling -- is declared
in `Suggests` and gated with `requireNamespace(..., quietly = TRUE)`
in the function bodies, so the package loads and checks without them.

## Vignettes

Vignettes are pre-built and shipped in `inst/doc/`; they rebuild
cleanly under `R CMD build`.

## DESCRIPTION authorship fields

`DESCRIPTION` carries both a modern `Authors@R:` field and an explicit
`Author:` field. The explicit `Author:` field is retained for
compatibility with the stricter author parsing introduced in R 4.6.0;
both fields agree on the maintainer and author identity.

## License

The package is licensed `AGPL-3` (`AGPL-3.0-or-later`) on both
language sides -- Python on PyPI and R here. The Linux-kernel adjuncts
shipped in the GitHub repository's `kernel-module/` and `daemon/`
directories remain `GPL-2.0-only` (a Linux kernel ABI requirement) and
are not part of the CRAN source tarball.

The deprecated `moirais` alias package is not part of this
submission; users install and upgrade through the canonical `morie`
package.

## AI-assistance disclosure

morie was developed with substantial assistance from Anthropic's
Claude family (used via Claude Code, Anthropic's official CLI agent)
and from Google DeepMind's Gemini 2.5 (Pro and Flash) on the Vertex AI
platform. Both providers extended research-credit programmes. Full
citations ship in the package's `CITATION.cff` and in the companion
papers' `.bib` files (bibkeys: `Anthropic2024ClaudeFamily`,
`Anthropic2024ClaudeCode`, `Bai2022ConstitutionalAI`,
`GoogleDeepMind2024Gemini`, `GoogleCloud2024VertexAI`). The author
retains full responsibility for the code, the methodology, the
empirical findings, and the published text; AI assistance accelerated
implementation, not authorship.
