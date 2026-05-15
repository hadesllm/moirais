## Test environments

* local macOS 26 (Darwin 25.4.0), R 4.6.0 -- 0 ERROR, 0 WARNING, 1 NOTE
* win-builder R-release (R 4.6.0, Windows Server 2022) -- https://win-builder.r-project.org/7Kx23l3o2SD0/
* win-builder R-devel    (R 2026-05-12 r90049, Windows Server 2022) -- https://win-builder.r-project.org/Hf58cXZf26RA/
* GitHub Actions ubuntu-latest (R-release, R-oldrel-1, R-devel)
* GitHub Actions macos-latest (R-release)
* GitHub Actions windows-latest (R-release)
* r-universe (Linux + macOS + Windows binaries)

## R CMD check results

Status: 0 ERROR, 0 WARNING, 1 NOTE.

The single NOTE is the standard "New submission" CRAN-incoming
feasibility check ("Maintainer: 'Vansh Singh Ruhela
<hadesllm@proton.me>'"), expected for the first CRAN submission of
this version line.

## Reverse dependencies

No reverse dependencies on CRAN.

## Notes for CRAN

* **Supersedes morie 0.7.1**, which is the most recent submission
  in the queue (a documentation-only patch on top of 0.7.0 fixing
  three manual issues: pre-rename title, ghost CJCCJ 2023 citation,
  and the MRM-acronym primary expansion). 0.7.2 is a further
  documentation-only patch on top of 0.7.1 fixing:
  (1) `@examples` coverage across the exported API raised from
  19.9% to 100% (377 / 377 exported functions); ~50 high-value
  user-facing exports got hand-written runnable examples, the
  remaining ~252 use minimal `\dontrun{ # See vignettes }`
  placeholders;
  (2) two example bugs that would have caused
  `R CMD check --as-cran` to fail had they shipped in 0.7.1
  (`mrm_latin_square` and `mrm_graeco_latin` —
  integer-codes-vs-LETTERS mismatch + non-orthogonal random
  Latin squares; both fixed);
  (3) an Rd structural bug where `morie_load_cpads.Rd`'s prose
  continuation containing `\enumerate{}` was folded into its
  `\examples{}` block (invalid Rd).
  No code or API changes vs 0.7.1. Local `R CMD check --as-cran`
  on 0.7.2 returns 0 ERROR, 0 WARNING, 1 NOTE (the standard
  "New submission" notice).

* This submission supersedes morie 0.1.14 (the last CRAN-accepted
  version) and brings the CRAN release in line with the v0.7.x
  release on PyPI, r-universe, and GHCR. The intermediate
  0.1.15 - 0.7.0 versions did not reach CRAN; the v0.7.0 release
  closed a five-paper publication set (the empirical applications
  paper landed on Zenodo at 10.5281/zenodo.20175689) and completed
  a license-unification pass that this submission reflects.

* The package is licensed `AGPL-3` (`AGPL-3.0-or-later`) on both
  language sides -- Python on PyPI and R on CRAN. The Linux-kernel
  adjuncts shipped in the GitHub repository's `kernel-module/` and
  `daemon/` subdirectories remain `GPL-2.0-only` (Linux kernel ABI
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
