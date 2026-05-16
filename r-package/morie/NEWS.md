# morie 0.9.0 — 2026-05-16

New: dataset availability auditing, more open-data sources, and
in-place self-update.

* **`check_datasets()` dataset auditor** — probes every entry in the
  dataset catalogue and reports which datasets are reachable and which
  need attention, classified by tier.
* **Statistics Canada ingest** — `morie.ingest.statcan` adds the
  Canadian Community Health Survey 2022 PUMF (StatCan 82M0013X) as the
  `cchs22` dataset, fetched on demand from the StatCan product page.
* **CIHI ingest** — `morie.ingest.cihi` adds five Canadian Institute
  for Health Information indicator data tables (hospital stays for harm
  caused by substance and alcohol use; youth integrated-youth-services
  access), fetched on demand from cihi.ca.
* **16 datasets wired to verified sources** — the Canadian Cannabis,
  Substance Use, Alcohol-and-Drugs, and Student survey PUMFs received
  verified open.canada.ca CKAN resource ids; the Toronto Police
  assault/homicide/shooting datasets and the Ontario SIU case data are
  now fetched through their existing scrapers. The catalogue went from
  33 to 49 reachable datasets.
* **New-version notification** — `import morie` performs a fail-silent,
  daily-cached check against PyPI and prints a one-line notice when a
  newer release exists. Opt out with `MORIE_NO_UPDATE_CHECK`.
  (Python interface.)
* **`morie update` command** — checks PyPI and, with confirmation,
  upgrades morie in place. (Python interface.)
* **CRAN fix** — the `morie_load_cpads` example is now wrapped in
  `\dontrun{}`, so `R CMD check --as-cran` no longer errors on the
  offline check farm.
* **Portable cache path** — the SQLite cache and on-demand fetched
  datasets now live in a per-user directory (`~/.cache/morie`, or
  `$XDG_CACHE_HOME`). A stale path calculation previously placed them
  outside any writable location; `MORIE_CACHE_DB` still overrides.
  Fixed identically on the R side, so the shared cache works.
* **`morie doctor --fix`** — the diagnostics command can now remediate
  failed checks: install missing Python packages, create the cache
  directory, and warn when a newer release is available. Plain
  `morie doctor` stays diagnostic-only. (Python interface.)
* **Missing-dataset recommendations** — when a dataset cannot be
  loaded, `load_dataset()` and `check_datasets()` now explain where it
  comes from — the CKAN portal, an on-demand fetcher, or the local
  path to place the file — via the new `dataset_recommendation()`
  helper.


# morie 0.8.0 — 2026-05-16

New: the fairness & disparity-audit subsystem (`morie.fairness`).

A subsystem for *auditing* risk-assessment, recidivism, and
predictive-policing systems for racial and other group disparities.
morie does not deploy such systems — it measures whether an existing
one encodes disparate treatment, so researchers and oversight bodies
can hold those systems accountable.

* **Six group-fairness metrics** — disparate impact ratio (the EEOC
  four-fifths rule), demographic parity gap, equalized odds, average
  odds difference, the Gini coefficient, and the composite Bias
  Amplification Score. Python and R, full parity.
* **Predictive-policing calibration audit** — `predpol_calibration_audit`
  ranks areas by predicted risk against realised outcomes and tests
  whether the disagreement tracks area demographics; paired with
  `predpol_score_disparity` and a city-agnostic `CityProfile` layer so
  the audit runs for any city. Python and R.
* **Multi-city temporal audit** — `predpol_temporal_audit` computes the
  four disparity metrics per (city, period) cell and surfaces temporal
  instability and cross-city divergence. Python and R.
* **Simulation framework** — a Noisy-OR patrol-detection model, a
  synthetic biased-crime-data generator, a JAX spatial GAN, and a
  CTGAN-style conditional tabular debiaser (the optional `morie[sim]`
  extra; JAX, not PyTorch, to stay lean).
* **Explainability (XAI) suite** — permutation importance (which flags
  protected features the model leans on), partial dependence,
  accumulated local effects, ceteris paribus, and sampling-based SHAP
  values; all model-agnostic.

The methods are clean-room reimplementations written from published
descriptions — IBM AIF360; the SciencesPo *Predictive-policing-Chicago*
project; Barman & Barman (arXiv:2603.18987); and the COMPAS audit in
pbiecek's *XAI Stories*. No third-party code was copied.

# morie 0.7.4 — 2026-05-16

Security patch.

* Fixed a regular-expression denial-of-service (ReDoS) vulnerability
  in the Ontario SIU scraper (`siu_fetch`). The index-page link
  parser used a repeated sub-pattern with `\s*` on both ends, which
  could cause catastrophic (exponential) backtracking on a maliciously
  crafted HTML page. The pattern is now linear-time; parsing of valid
  SIU index pages is unchanged. (CodeQL `py/redos`, high severity.)
* `User-Agent` strings across the data-ingestion modules were stale
  (`morie/0.2.0`–`morie/0.6.1`) and are now aligned to the release
  version.
* No API changes.

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
