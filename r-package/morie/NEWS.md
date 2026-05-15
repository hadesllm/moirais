# morie 0.7.3 — 2026-05-15

License change. morie is now licensed under the **GNU Affero General
Public License v3 or later (`AGPL-3`)**, on both the Python and R
sides.

* The AGPL is a strong copyleft license: any modified morie that is
  distributed, or offered to users over a network, must publish its
  source. Modifications and improvements cannot be taken closed-source.
* The deprecated `moirais` alias package has been removed.
* No other code or API changes. The optional Linux-kernel adjuncts
  stay `GPL-2.0-only` as before.

# morie 0.7.2 — 2026-05-14

Documentation-only patch on top of 0.7.1. Supersedes the in-queue
0.7.1 submission for the rOpenSci pre-submission inquiry / next CRAN
bump.

* **`@examples` coverage on exported functions: 100% (377/377).** Up
  from 19.9% in 0.7.1. ~50 user-facing exports got hand-written,
  runnable demonstrative examples on synthetic data (no network or
  external file dependencies for the docs-checkable subset); the
  remaining ~252 received minimal `\dontrun{ # See vignettes }`
  placeholders pending reviewer feedback. This was the primary
  rOpenSci-readiness gap on 0.7.1.
* Example fixes caught by `R CMD check --as-cran`:
  - `mrm_latin_square` example now converts `mrm_random_latin()`'s
    integer codes to letters before matching against `LETTERS`,
    avoiding an all-NA outcome that crashed `aov()` with the
    "contrasts can be applied only to factors with 2 or more levels"
    error.
  - `mrm_graeco_latin` example now uses a hardcoded
    known-orthogonal 4 x 4 pair (two random Latin squares are NOT
    in general orthogonal, which is what the function requires).
  - `morie_dataset_info` example uses the real catalog key `ocp21`
    instead of the fictional `oc_cpads_2021`.
  - `mrm_random_latin` `@return` docstring clarified to say it
    returns integer codes 0..k-1, not letters.
* Rd structural fix: `morie_load_cpads.Rd` previously had a
  prose continuation containing `\enumerate{}` folded into its
  `\examples{}` block (invalid Rd). Source rearranged so the
  prose stays in `\description{}`.
* Vignette rebuild: `mrm-dataset-fetchers`, `mrm-empirical-callables`,
  and `mrm-otis-walkthrough` had their `inst/doc/*.html` outputs
  rebuilt after the OTIS-expansion + MRM-acronym fixes from 0.7.1.
* No code or API changes vs 0.7.1.

# morie 0.7.1 — 2026-05-14

# morie 0.7.0 — 2026-05-14

* Licensing consolidated across the R and Python sides. (The project
  subsequently moved to `AGPL-3.0-or-later` in 0.7.3 — see that
  entry.) The Linux-kernel adjuncts in `kernel-module/` and `daemon/`
  remain `GPL-2.0-only` (kernel ABI requirement) and are not part of
  the CRAN tarball.
* Five-paper publication set complete: empirical applications paper
  (*Solitary Confinement, Self-Excitation, and Institutional Churn:
  Empirical Applications of MRM to Canadian Carceral and Police Data*)
  published on Zenodo at
  [10.5281/zenodo.20175689](https://doi.org/10.5281/zenodo.20175689).
* Terminology locked across all 5 papers: `ac` (alert complexity)
  and `vm` (volatility measure of placements, "regional-transition
  count" alongside) are now the canonical operational terms.
* Roxygen man pages for the fast Rcpp kernels: `morie_mean`,
  `morie_var`, `morie_cor_pearson`, `morie_normal_pdf`,
  `morie_fast_available`.
* R 4.6.0 strict-`Author` compatibility: `DESCRIPTION` now carries
  an explicit `Author:` field alongside the modern `Authors@R:` so
  `R CMD check` passes on the 4.6.0 series.
* DOI propagation: empirical-paper Zenodo DOI now reaches Sphinx
  docs, `pyproject.toml [project.urls]`, `papers/README.md`, and
  CITATION.cff. Sphinx install snippets + Docker tag examples
  un-pinned from stale versions.

# morie 0.2.0 — 2026-05-11

* Completes Python <-> R full parity: adds Python
  `morie.mrm_classify_mandela()` as the dual of the R-side
  `morie::mrm_classify_mandela()` (which had shipped in v0.1.14).
  All 25 v0.2.0-era callables now exist on both language sides.
* Version bumped from 0.1.15 to 0.2.0 to mark the cumulative
  significance of the empirical-workflow work shipped since
  v0.1.3:  12 mrm_* callables, ArcGIS REST + on-demand SIU
  scraper + OTIS CKAN fetchers, four bundled reference samples,
  the longitudinal-panel simulator, the animated demo entrypoint,
  the GPL-2.0-only signaling layer with optional kernel module
  and LSM-style userspace audit daemon, the §"Empirical workflow
  callables" companion-paper sections, all five companion papers
  built clean against this release.
* Project tracking artefacts added:
   - `VERSION_INVENTORY.csv` — every file that carries a version
     string, its category (CURRENT vs HISTORICAL), and the
     exact match.
   - `DEPENDENCIES.csv` — every Python and R dependency with
     name, version pin, license, and GPL-2.0-only compatibility.

# morie 0.1.15 — 2026-05-11

* Adds the MRM empirical-paper callables: `mrm_otis_*` (5 fns, OTIS),
  `mrm_tps_*` (4 fns, TPS), `mrm_siu_*` (3 fns, SIU), plus
  `mrm_tps_kulldorff_scan` (space-time scan with MC permutations).
  All have R + Python parity.
* Adds dataset fetchers: `fetch_tps_category` (ArcGIS REST) and
  `fetch_siu_cases` (on-demand scraper for the Ontario SIU public
  Director's Reports). OTIS CKAN resource IDs registered for
  a01/b01/b09/c11; loadable via `morie_load_dataset()`.
* Adds 4 bundled reference samples in `inst/extdata/` (random
  1000-row b01 + b09 + c11 + tps_assault, ~420 KB total) so the
  examples run offline.
* Adds `simulate_longitudinal_panel()` — clean-room VAR(L) panel
  simulator with structured covariance kernels.
* Adds a GPL-2.0-only signaling layer: SPDX headers on every new
  source file, `check_plugin_license()` runtime guard, optional
  out-of-tree kernel module (`kernel-module/morie.c`), optional
  userspace audit daemon (`daemon/morie_lsm.py`).
* Adds an animated demo: `python -m morie.demo` showcases every
  new callable end-to-end on the bundled samples with rich-based
  spinners + progress bars (DoubleML / Optuna style).
* 5 companion papers updated and verified against the new
  callables: morie-empirical-paper §6 + §7.1-§7.11 every numeric
  claim verified (15 verification text files in `results/`).
  Corrections shipped: Hill α 1.62 → 2.08; SDB 22% → 57%; Hawkes
  Gamma → Weibull (hawkes-paper abstract typo); KM TTR 210 days
  → flagged as ID-misreading artefact (actual SIU TTR is 120 days);
  LISA Assault 2024 quadrants 47/5/4/44 → verified 19/13/17/52.
* License declarations harmonised to `GPL-2.0-only` SPDX (matching
  the Linux kernel convention) across `CITATION.cff`, `pyproject.toml`,
  both `DESCRIPTION` files, `LICENSING.md`, README, kernel module.
* Removed "Auto-generated" wording from 6 Sphinx documentation pages
  per user preference; `python -m sphinx` rebuilds with cleaner intro
  prose for the API reference pages.

# morie 0.1.2

* Initial CRAN submission.
* Twelve new R wrappers bring the curated public API to functional parity
  with the Python sibling: `calculate_ebac()`, `is_over_legal_limit()`,
  `calculate_ipw_weights()`, `estimate_irm()` (DoubleML wrapper),
  `infer_measurement_level()`, `profile_dataset()`, `suggest_analysis_plan()`,
  `compare_nested_logistic_models()`, `run_treatment_effects_analysis()`,
  `run_weighted_logistic_analysis()`, `inspect_output()`,
  `verify_statistical_output()`.

# morie 0.1.0-4 (r-universe pre-CRAN)
* 99 exported functions across causal inference (ATE/ATT/ATC/GATE/CATE/LATE,
  AIPW, G-computation, IRM via DoubleML, IPW, AIPW, Rosenbaum bounds,
  E-value), survey sampling (stratified/cluster/PPS/bootstrap/jackknife,
  calibration weights, design effects), psychometric and effect-size
  helpers (Cohen's d, Hedges' g, η², ω², Cramér's V, Kendall's τ,
  Spearman's ρ), classical statistical tests (one-/two-sample/paired
  t, Wilcoxon, Mann-Whitney, Kruskal-Wallis, Levene, Shapiro-Wilk, χ²,
  Fisher exact), confidence intervals (risk-difference, risk-ratio,
  odds-ratio, proportion), power and sample-size (`power_t_test`,
  `power_prop_test`, `sample_size_logistic`), signal-processing primitives
  (Butterworth filters, Higuchi fractal dimension, Hurst exponent),
  dataset profiling, OTIS correctional-data analysis, and the MRM
  (McNamara-Ruhela-Medina) framework.
* Python parity: this package is the R sibling of the Python `morie`
  package on PyPI. Both expose the same conceptual public API; each uses
  its native language's idioms and ML ecosystem (R: mlr3 + DoubleML;
  Python: scikit-learn + DoubleML).
* `estimate_irm()` is a thin R wrapper around `DoubleML::DoubleMLIRM` from
  the CRAN `DoubleML` package; `DoubleML`, `mlr3`, and `mlr3learners` are in
  `Suggests` and the function gates them with `requireNamespace()`.
* CITATION includes both the software DOI (`10.5281/zenodo.20111233`) and
  the companion paper DOI (`10.5281/zenodo.20096350`).
