## Test environments

* local macOS 26 (Darwin 25.4), R 4.5, R-devel
* GitHub Actions, ubuntu-latest (R release, R oldrel-1, R devel)
* GitHub Actions, macos-latest (R release)
* GitHub Actions, windows-latest (R release)
* r-universe (Linux + macOS + Windows binaries)

## R CMD check results

Status: OK. 0 errors, 0 warnings, 0 notes.

## Reverse dependencies

This is a new release. No reverse dependencies.

## Notes for CRAN

* This is the first stable CRAN submission of `morie` (version 0.3.0).
* The R package is licensed GPL-2.0-only to match the R ecosystem
  convention; the sibling Python package on PyPI is dual-licensed
  MIT OR Apache-2.0 (the Rust ecosystem convention).
* Optional dependencies (`spdep`, `gstat`, `survival`, `forecast`,
  `xgboost`, `gbm`, `caret`, etc.) are declared in Suggests and gated
  by `requireNamespace(..., quietly = TRUE)` in function bodies.
* Vignettes are pre-built and shipped in `inst/doc/`.
* The deprecated `moirais` alias package is not part of this
  submission; users get a clean upgrade path via the `morie` package.
