# morie roadmap

This document lists what's shipped and what's planned. Items below v1.0.0 are pre-alpha point releases — see [INSTALLATION.md](INSTALLATION.md) for the stability posture.

## Shipped

### v0.7.0 — five-paper completion + license migration + DOI propagation (current, 2026-05-14)

The release that closes the publication set, migrates the R package off `GPL-2.0-only`, and propagates the new empirical-paper DOI everywhere downstream consumers look:

- **License migrated to `MIT OR Apache-2.0` (both language sides)** — the R package switches from `GPL-2.0-only` to `Apache License (== 2) | MIT + file LICENSE` (CRAN form). Python is unchanged. The two optional Linux-kernel adjuncts (`kernel-module/morie.c` and `daemon/morie_lsm.py`) stay `GPL-2.0-only` (kernel ABI requirement) and are explicitly NOT part of the wheel or CRAN tarball. 279 R source-file SPDX headers swept.
- **Empirical applications paper now public on Zenodo** — *Solitary Confinement, Self-Excitation, and Institutional Churn: Empirical Applications of MRM to Canadian Carceral and Police Data*, [10.5281/zenodo.20175689](https://doi.org/10.5281/zenodo.20175689). Five-paper set (MRM framework, Hawkes methodology, morie-R software, morie-Python software, empirical applications) is now complete.
- **`ac` / `vm` terminology locked across all 5 papers** — `ac` (alert complexity) and `vm` (volatility measure of placements, "regional-transition count" alongside) are now the canonical operational terms.
- **DOI + version sweep across all surfaces** — empirical-paper DOI now reaches `docs/source/index.rst` Sphinx badges, `pyproject.toml [project.urls]`, `papers/README.md`, and `CITATION.cff`. Also: `Dockerfile` un-pinned from `0.2.0` → `0.7.0`; Sphinx install snippets un-pinned from `0.4.12` → `0.7.0`; `papers/README.md` "in preparation" → published DOI for the empirical paper; v0.5.0 version stamp note → v0.7.0.
- **R-package roxygen man pages for the fast Rcpp kernels** — `morie_mean`, `morie_var`, `morie_cor_pearson`, `morie_normal_pdf`, `morie_fast_available` now ship with proper Rd documentation alongside the C++ exports.
- **R 4.6.0 strict-`Author` compatibility** — `DESCRIPTION` carries an explicit `Author:` field alongside the modern `Authors@R:` so `R CMD check` passes on the 4.6.0 series (which no longer auto-derives `Author` from `Authors@R` during source-tree checks).

### v0.5.0 — fresh-user release (2026-05-13)

The release that makes the toolkit usable by a non-programmer:

- **No-code CLI** — `morie tutorial` (interactive walkthrough), `morie pull tps-major --year 2024`, `morie run-module power-design`, `morie explain power_summary.csv`, `morie cheatsheet`, `morie generate-template`.
- **Synthetic CPADS / OTIS / TPS / SIU datasets** bundled in-wheel so a first analysis works with no manual file downloads.
- **Open-data ingestors** — `morie ingest ckan` (open.canada.ca and any CKAN portal), `morie ingest tps` (ArcGIS layers), `morie ingest siu` (Director's-report PDFs).
- **Schema-agnostic loader** — datasets with non-canonical column names flow through morie modules without renaming; `morie.schema.infer_mapping(your_df, canonical=...)` does the mapping.
- **Nine-language CLI** — EN / FR / ES / DE / ZH / PT / JA / AR / HI via `MORIE_LOCALE=<code>`.
- **Five install channels verified** end-to-end on Mac + Raspberry Pi: curl one-liner, `pip install morie`, `brew tap hadesllm/morie && brew install morie`, `docker run ghcr.io/hadesllm/morie:0.5.0`, R via r-universe.
- **First-paper template** — `morie generate-template --module <name> --out my-paper.md` writes a JSS-style methods + results scaffold pre-filled with BibTeX entries for the four companion papers.
- **Accessibility** — `NO_COLOR=1` and pipe-detection auto-degrade to plain text for screen readers.

### v0.1 – v0.4 (April-May 2026, foundations + hardening)

| Series | What landed |
|---|---|
| v0.1.x | First public release; rename from MOIRAIS in v0.1.3; 23 analysis modules; 275 textbook-derived R callables; OTIS / SIU / TPS ingestion; first c11 Mandela classifier. |
| v0.2.x | `morie.entheo` DMT-imaging opt-in (Timmermann dataset); deterministic-seed plumbing for R/Python bit-for-bit parity; PEP 562 lazy-loading drops cold import from 123 s to 1.7 s. |
| v0.3.x | Per-component licensing first finalised (Python MIT OR Apache-2.0, R GPL-2.0-only at the time — later revised to MIT + Apache in v0.7.0); R/Python parity verification; 5 JSS-style companion papers drafted. |
| v0.4.x | `anova_oneway` backwards-compat alias; `gibbons_chakraborti` canonical naming (Gibbons & Chakraborti, 2003 §2.11); Homebrew tap; GHCR container made public; auto-tag-on-version-bump CI; Windows install in CI matrix; `\cite{}` → `\citep{}` JSS citation cleanup. |

## Planned

### v0.6.0 — performance experiments

Optional, opt-in JIT-compile and C++ paths for the slowest numerical kernels. Won't speed CI; will speed end-user runtime of repeated calls.

**Shipped:**
- `morie.fast` public namespace exposing the JIT-decorated kernels (`normal_pdf`, `normal_logpdf`, `mean_jit`, `var_jit`, `std_jit`, `cor_pearson_jit`, `euclid_dist_jit`) + a `jit_if_available` decorator users can apply to their own loops. Numerically identical to scipy/numpy whether or not Numba is installed; verified to ≤ 5.55e-17 max error.
- `pip install morie[fast]` extra pulls Numba on py ≤ 3.14; on py ≥ 3.15 the install falls back to pure-numpy and `morie.fast.is_jit_available()` returns False.
- `ci-numba-bench.yml` workflow runs nightly + on-demand, benchmarks the JIT path against scipy / numpy baselines, asserts numerical agreement.

**In-flight:**
- Sprinkling `@jit_if_available` on the Hawkes log-likelihood inner loop in `tps_hawkes_advanced.py`, the IPW weight loop in `causal.py`, the bootstrap resampler in `btsrp.py`, and the MRM 10-estimator ensemble loop in `mrm_design.py`. Target: 2-5× speedup on Hawkes-paper reproducibility runs.

**Planned:**
- **Rcpp / RcppArmadillo for R hot kernels** — translate the slowest pure-R kernels (rugarch glue, per-row MRM ensemble loop, fuzzy-rule expansions) via the `LinkingTo: Rcpp` pattern. Separate `ci-rcpp-bench.yml` workflow for nightly perf check; main `R CMD check` stays pure-R to keep CI lightweight.

### v0.6.1 — methodological + dataset expansion (SHIPPED, 2026-05-13)

Three new modules drawn from Laniyonu et al.'s policing-and-corrections methodology, five new MRM primitives, three US/Canadian crime-data integrations, and a forensics-dataset bundle.

#### New analysis modules

- `morie.laniyonu.gentrification_policing(stops_df, crime_df, demand_df, acs_df, geom_df, estimator="sdm", ...)` — Spatial Durbin Model decomposing direct / indirect / total effects of gentrification on stop-and-frisk rates. Reproduces Laniyonu (2018) *Urban Affairs Review* 54(5):898–930.
- `morie.laniyonu.smi_force_disparity(force_df, survey_df, acs_df, estimator="negbin_hier_bayes", ...)` — Bayesian hierarchical negative-binomial with synthetic small-area-estimated exposure offset. Reproduces Laniyonu & Goff (2021) *BMC Psychiatry* 21(1):500 — relative risk of force for persons with serious mental illness = 11.6× the non-SMI baseline.
- `morie.laniyonu.actuarial_risk_disparity(intake_df, outcome_df, model="ordinal_logit_bayes", threshold_specific=True, ...)` — Bayesian cumulative-logit with threshold-varying race / gender coefficients. Reproduces O'Connell & Laniyonu (2025) *Race & Justice* 15(3):428–453 — bias concentrated at the low→medium ordinal cutoff in Canadian federal-prison risk assessments.

#### New MRM (Multilevel Reconciliation Methodology) primitives

The Laniyonu papers contribute five reusable identification patterns that any morie module can call:

1. `mrm.primitive.gentrification_panel` — categorical baseline-conditional gentrification coding (ineligible / eligible-no-change / gentrified) from baseline-marginalisation + top-tercile growth rules. The "eligible-but-did-not-gentrify" tract is the cleanest comparator.
2. `mrm.primitive.spatial_spillover_decomposition` — Spatial Durbin / SAR direct / indirect / total decomposition with the Moran's I + Robust LM diagnostic ladder.
3. `mrm.primitive.synthetic_area_exposure` — Small-area estimation as exposure offset. Generalises beyond SMI to homelessness, LGBTQ, undocumented, or any "rate per hidden subpopulation" estimand.
4. `mrm.primitive.threshold_specific_ordinal` — Bayesian cumulative-logit with race / gender coefficients allowed to vary by ordinal threshold.
5. `mrm.primitive.score_net_residual` — Two-stage score-then-audit; residual race effect on downstream outcome (parole, housing) net of the assessment score = disparate treatment signal.

#### New datasets

- `morie.datasets.chicago_crime()` — Chicago open-data crime feed (Socrata API).
- `morie.datasets.nyc_stop_and_frisk()` — NYPD SQF microdata.
- `morie.datasets.la_crime()` — LAPD open-data feed.
- `morie ingest bigquery --project bigquery-public-data --dataset chicago_crime --table crime --year 2024` — Generic BigQuery public-data integration via ADC.
- Forensics: `morie.datasets.nist_rds()`, `nibrs()`, `namus_missing_persons()`, `icitap_*()`.
- Three toy bundles for offline replication: `load_gentrification_nyc()` (11k tract-years), `load_smi_force_synthetic()` (28k events + 5.5k NCS-R survey + ACS), `load_csc_risk_synthetic()` (50k CSC sentence-level mock).

#### New citations across all four refs.bib files

- `Laniyonu2018CoffeeShops` — doi:10.1177/1078087416689728
- `LaniyonuGoff2021BMCPsych` — doi:10.1186/s12888-021-03510-w
- `OConnellLaniyonu2025RaceJustice` — doi:10.1177/21533687231153993

### v1.0.0 — first alpha milestone

Pre-alpha (v0.x) is the cumulative point-release work above. The first alpha label belongs to v1.0.0, by which point: the user-visible CLI surface is API-stable, the MRM primitives are individually papered, the published findings are independently reproduced by at least one external collaborator, and the dependency-graph hardening pass survives a CRAN re-submission round-trip.

No timeline commitment on v1.0.0 — it ships when those bars are met, not on a calendar.

## Out of scope (won't ship in v0.6.x)

- C++ rewrite of any core analysis module (R/Python Numba/Rcpp opt-ins are the limit).
- Web UI / browser-based front end (CLI + Python API are the only supported surfaces).
- Non-Canadian carceral / non-US-policing domain extensions — those wait until v0.7.x and a domain-expert collaboration.
- Real-time streaming ingestion — every adapter is point-in-time pull-only.

## How to suggest changes

Open an issue at <https://github.com/hadesllm/morie/issues> with the tag `roadmap:` followed by the version (e.g. `roadmap: v0.6.1`). The maintainer triages monthly.
