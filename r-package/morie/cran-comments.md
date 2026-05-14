## Test environments

* local macOS 26 (Darwin 25.4.0), R 4.6.0
* GitHub Actions, ubuntu-latest (R release, R oldrel-1, R devel)
* GitHub Actions, macos-latest (R release)
* GitHub Actions, windows-latest (R release)
* r-universe (Linux + macOS + Windows binaries)
* win-builder (R-release + R-devel)

## R CMD check results

Status: 0 ERROR, 0 WARNING, 1 NOTE.

The single NOTE is the standard "New submission" CRAN-incoming
feasibility check ("Maintainer: 'Vansh Singh Ruhela
<hadesllm@proton.me>'"), expected for a version-bumped resubmission
of a package whose last accepted version was 0.1.14.

## Reverse dependencies

No reverse dependencies on CRAN.

## Notes for CRAN

* This submission is `morie` version 0.7.0. It supersedes morie
  0.1.14 (the last CRAN-accepted version) and brings the CRAN
  release in line with the v0.7.0 release on PyPI, r-universe,
  and GHCR. The intermediate 0.1.15 - 0.6.1 versions did not reach
  CRAN; the v0.7.0 release closes a five-paper publication set
  (the empirical applications paper landed on Zenodo at
  10.5281/zenodo.20175689) and completes a license-unification
  pass that this resubmission reflects.

* The package is dual-licensed `Apache License (== 2) | MIT + file
  LICENSE` (CRAN form). Both language sides of the project --
  Python on PyPI and R on CRAN -- now follow the Rust-ecosystem
  dual-license convention. The Linux-kernel adjuncts shipped in
  the GitHub repository's `kernel-module/` and `daemon/`
  subdirectories remain `GPL-2.0-only` (Linux kernel ABI
  requirement) and are not part of the CRAN source tarball.

* The package's `DESCRIPTION` carries both a modern `Authors@R:`
  field and an explicit `Author:` field. The explicit `Author:`
  field is retained for compatibility with the strict author
  parsing introduced in R 4.6.0; both fields agree on the
  maintainer and author identity.

* Optional dependencies (`spdep`, `gstat`, `survival`, `forecast`,
  `xgboost`, `gbm`, `caret`, etc.) are declared in `Suggests` and
  gated by `requireNamespace(..., quietly = TRUE)` in function
  bodies. There are no hard runtime dependencies beyond the base
  + recommended set plus `Rcpp`.

* Vignettes are pre-built and shipped in `inst/doc/`. They rebuild
  cleanly under `R CMD build`.

* The deprecated `moirais` alias package is not part of this
  submission; users upgrade through the canonical `morie` package.

* AI-assistance disclosure: morie was developed with substantial
  assistance from Anthropic's Claude family (used via Claude Code,
  Anthropic's official CLI agent) and from Google DeepMind's
  Gemini 2.5 (Pro and Flash) on the Vertex AI platform. Both
  providers extended research-credit programmes. Full citations
  ship in the package's `CITATION.cff` and in the companion papers'
  `refs.bib` files (bibkeys: `Anthropic2024ClaudeFamily`,
  `Anthropic2024ClaudeCode`, `Bai2022ConstitutionalAI`,
  `GoogleDeepMind2024Gemini`, `GoogleCloud2024VertexAI`). The
  author retains full responsibility for the code, the methodology,
  the empirical findings, and the published text; AI assistance
  accelerated implementation, not authorship.
