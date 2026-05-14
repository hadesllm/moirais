# MORIE

**Multi-domain Open Research and Inferential Estimation**

> *Renamed from MOIRAIS in v0.1.3. The `moirais` Python module and the `moirais` R package remain available as deprecated aliases — `import morie` / `library(morie)` is the canonical entry point going forward.*

A multi-domain scientific computing toolkit (Python and R) for observational inference, with sociolegal, signal-processing, cryptographic, spatial-statistics, statistical-physics, and psychometrics modules. Hosts the MRM framework as a primary application for Canadian carceral, police, and oversight data analysis.

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT_OR_Apache--2.0-3776ab.svg)](https://github.com/hadesllm/morie/blob/main/LICENSE-MIT)
[![PyPI version](https://img.shields.io/pypi/v/morie.svg)](https://pypi.org/project/morie/)
[![r-universe](https://img.shields.io/badge/r--universe-hadesllm-276DC3)](https://hadesllm.r-universe.dev/morie)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DOI · morie R](https://img.shields.io/badge/DOI%20%C2%B7%20morie%20R-10.5281%2Fzenodo.20111233-0d9488?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.20111233)
[![DOI · morie Python](https://img.shields.io/badge/DOI%20%C2%B7%20morie%20Python-10.5281%2Fzenodo.20096350-7c3aed?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.20096350)
[![MRM paper](https://img.shields.io/badge/MRM_paper-10.5281%2Fzenodo.20096075-15803d?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.20096075)
[![Hawkes paper](https://img.shields.io/badge/Hawkes_paper-10.5281%2Fzenodo.20102198-be123c?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.20102198)
[![Empirical paper](https://img.shields.io/badge/Empirical_paper-10.5281%2Fzenodo.20175689-1a73e8?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.20175689)

> ⚠️ **Pre-alpha (v0.x).** MORIE is in pre-alpha. The first alpha milestone is **v1.0.0**; everything before that is point-releases of pre-alpha code. APIs may shift, datasets may move, and findings may be refined between minor versions. Paper sources are at [`papers/`](https://github.com/hadesllm/morie/tree/main/papers) (LaTeX); compiled PDFs are on Zenodo via the DOI badges above.

## Installation

> Full step-by-step install guide with platform-specific notes (PEP 668 on Debian, python 3.13 segfault on Raspberry Pi OS, etc.) is at **[INSTALLATION.md](https://github.com/hadesllm/morie/blob/main/INSTALLATION.md)**.

### Recommended — one-liner (Linux / macOS / WSL)

The simplest path. Bootstraps everything you need (Python via `uv`, a managed venv, the morie wheel) — works even if you have **no Python and no `pip` installed**.

```bash
curl -fsSL https://hadesllm.github.io/morie/install.sh | bash
```

Or, with R alongside Python:

```bash
curl -fsSL https://hadesllm.github.io/morie/install.sh | bash -s -- --auto
```

After install, `~/.local/bin/morie` is a thin shim into the managed venv at `~/.venvs/morie`. Full install instructions, channel comparison, and platform-specific notes are at **[hadesllm.github.io/morie/#quick-start](https://hadesllm.github.io/morie/#quick-start)**.

### Python — Homebrew (macOS / Linuxbrew)

```bash
brew tap hadesllm/morie
brew install morie
```

The tap repo is [`hadesllm/homebrew-morie`](https://github.com/hadesllm/homebrew-morie). It pulls morie's source distribution from PyPI and bundles a self-contained `python@3.12` venv — no system Python required.

### Python — PyPI (manual; requires `pip` already installed)

```bash
pip install morie
```

> **Heads-up:** modern Debian / Ubuntu / Raspberry Pi OS forbid `pip` outside virtual environments (PEP 668), and the system `python3` on Raspberry Pi OS 13 segfaults on importing the SciPy stack. If `pip install morie` errors or `import morie` segfaults, use the one-liner above instead — it handles both cases automatically.

### Python — Docker (no local dependencies)

```bash
# Latest stable
docker run --rm ghcr.io/hadesllm/morie:latest morie --help

# Pin to a specific version (recommended for reproducibility)
docker run --rm ghcr.io/hadesllm/morie:0.7.0 morie --help
```

Multi-arch image published on every release with both versioned and `:latest` tags. Requires only Docker — no Python, no pip.

### R — CRAN (when available) or r-universe

```r
# Stable from CRAN (when listing is live)
install.packages("morie")

# Nightly binary builds (recommended while CRAN listing is rolling out)
install.packages(
  "morie",
  repos = c(
    hadesllm = "https://hadesllm.r-universe.dev",
    CRAN     = "https://cloud.r-project.org"
  )
)
```

## Quick start

```python
import morie

# Load a built-in dataset
df = morie.load_dataset("otis-2025")

# Run an MRM module on OTIS data
from morie.otis_all_analyze import analyze_a01_mrm
result = analyze_a01_mrm(df)
print(result)
```

## What's new in v0.7.0

- **License migrated to `MIT OR Apache-2.0` on both language sides** — the R package switches from `GPL-2.0-only` to `Apache License (== 2) | MIT + file LICENSE` (CRAN form). Python is unchanged (already `MIT OR Apache-2.0`). The two optional Linux-kernel adjuncts (`kernel-module/` and `daemon/`) stay `GPL-2.0-only` because the kernel ABI requires it; they are not part of the wheel or CRAN tarball.
- **Empirical applications paper published** — *Solitary Confinement, Self-Excitation, and Institutional Churn: Empirical Applications of MRM to Canadian Carceral and Police Data* on Zenodo at [10.5281/zenodo.20175689](https://doi.org/10.5281/zenodo.20175689). Five-paper publication set now complete.
- **`ac` / `vm` terminology locked across all 5 papers** — `ac` (alert complexity) and `vm` (volatility measure of placements, "regional-transition count" alongside) are now the canonical operational terms.
- **DOI + version propagation sweep** — empirical-paper DOI now reaches Sphinx index, `pyproject.toml [project.urls]`, `papers/README.md`, and CITATION.cff. Sphinx install snippets, Docker tag examples, and the in-tree `papers/README.md` were also un-pinned from stale versions.
- **R-package roxygen docs for fast Rcpp kernels** — `morie_mean`, `morie_var`, `morie_cor_pearson`, `morie_normal_pdf`, `morie_fast_available` ship with Rd man pages.
- **R 4.6.0 compatibility** — `DESCRIPTION` carries an explicit `Author:` field alongside `Authors@R:` so `R CMD check` passes on the strict 4.6.0 build.

## What's new in v0.6.1

- **Three replication modules from Laniyonu et al.** — `morie.laniyonu.gentrification_policing()` (Spatial Durbin replication of Laniyonu 2018 *UAR* — gentrification spillover on NYPD SQF), `morie.laniyonu.smi_force_disparity()` (Bayesian-style hierarchical neg-binomial replication of Laniyonu & Goff 2021 *BMC Psych* — police force on persons with serious mental illness), `morie.laniyonu.actuarial_risk_disparity()` (cumulative-logit replication of O'Connell & Laniyonu 2025 *Race & Justice* — Canadian federal-prison risk-assessment bias).
- **Five reusable MRM identification primitives** — `mrm.primitive.gentrification_panel`, `spatial_spillover_decomposition`, `synthetic_area_exposure`, `threshold_specific_ordinal`, `score_net_residual`. The building blocks every future module composes.
- **US + Canadian crime-data adapters** — `morie.datasets.chicago_crime()`, `nyc_stop_and_frisk()`, `bigquery()` (lazy Google-Cloud BigQuery), plus `nibrs()` (FBI Crime Data Explorer), `namus_missing_persons()`, `nist_rds()` (NIST Reference Datasets catalog).
- **Toy bundles for every new dataset** — Chicago crime (50 rows), NYC SQF (40 rows), NIBRS (30 rows), NamUs (20 rows), NIST RDS (10 rows). `offline=True` works on every loader.
- **`morie.fast` opt-in JIT acceleration surface** — drop-in JIT-compiled kernels (`normal_pdf`, `cor_pearson_jit`, `bootstrap_mean_jit`, `trimmed_ipw_weights_jit`, …) + a `jit_if_available` decorator. `pip install morie[fast]` activates Numba; without it, kernels run as pure-numpy. Numerically identical to scipy/numpy (max diff ≤5.55e-17).
- **`ci-numba-bench.yml`** nightly benchmark workflow comparing JIT vs non-JIT paths on every release.
- **Three new BibTeX entries** added to all 4 paper bibliographies: Laniyonu (2018), Laniyonu & Goff (2021), O'Connell & Laniyonu (2025).
- **Lazy-import fix** in `morie.ingest.__init__` — PEP 562 `__getattr__` for BigQuery uses `importlib.import_module` to avoid the infinite-recursion trap that `from . import bigquery` would create.

## What's new in v0.5.0

- **Any-dataset support** — bring your own column names. `morie.schema.infer_mapping(your_df, canonical=...)` fuzzy-matches your columns onto morie's canonical schema; pass the dict to `apply_mapping` and your data flows through every module without renaming. CLI users get `morie run-module ... --columns my_wt:weight,drinks_yn:alcohol_past12m`.
- **9-locale CLI** — `MORIE_LOCALE=es|de|zh|pt|ja|ar|hi morie ...` plus the existing EN + FR. Methodology docs stay English; CLI surface is multilingual.
- **No-code dataset shortcuts** — `morie pull tps-major --year 2024 --out file.csv` writes the entire Toronto Police "Major Crime" feed to disk in one line. No Python, no API URLs, no SQL. Also: `morie pull tps-shootings`, `morie pull tps-homicide`, `morie pull cpads`, `morie pull otis-a01-toy`, `morie pull siu-toy`, `morie pull tps-layers`.
- **[`TUTORIAL.md`](https://github.com/hadesllm/morie/blob/main/TUTORIAL.md)** — your first analysis, no Python knowledge required. Copy-paste five commands and you have 13 CSVs explained.
- **Python facade** — `import morie.datasets as md; df = md.tps_major_crime(year=2024)` for users who want to script.
- **Open-data adapters** — `morie ingest ckan/tps/siu` pulls feeds from CKAN portals (open.canada.ca, data.gov.uk, etc.), Toronto Police Service ArcGIS layers, and Special Investigations Unit director's-reports directly into pandas. See `morie.ingest.{ckan,tps,siu}`.
- **Synthetic CPADS bundled** — `morie run-module power-design` works on a fresh install with no manual download; emits a clear "synthetic data" warning so toy outputs aren't mistaken for real findings.
- **[`INSTALLATION.md`](https://github.com/hadesllm/morie/blob/main/INSTALLATION.md)** walkthrough covering all 5 install channels with platform-specific notes (PEP 668 on Debian, python 3.13 segfault on Raspberry Pi OS, Windows).
- **[`papers/`](https://github.com/hadesllm/morie/tree/main/papers)** allowlisted JSS paper sources in-tree (5 papers; no emails or drafts).
- Sphinx **"Edit on GitHub"** link in the sidebar so readers can suggest doc changes in one click.
- `anova_oneway` backwards-compat alias + `gibbons_chakraborti` rename (from v0.4.14, carried forward).

## Documentation

Full documentation is at [hadesllm.github.io/morie](https://hadesllm.github.io/morie/).

## Citation

If you use morie in your research, please cite **both software papers** (R and Python)
and, where applicable, the **MRM framework paper**, the **Hawkes methodology paper**,
and the **empirical applications paper**.

```
# Software paper — R (also the R package source on Zenodo)
Ruhela, V. S. (2026). morie: Multi-domain Open Research and Inferential
Estimation in R (v0.7.0). Zenodo.
https://doi.org/10.5281/zenodo.20111233

# Software paper — Python (also the Python package source on Zenodo)
Ruhela, V. S. (2026). morie: Multi-domain Open Research and Inferential
Estimation in Python (v0.7.0). Zenodo.
https://doi.org/10.5281/zenodo.20096350

# MRM framework paper (theoretical foundations)
Ruhela, V. S. (2026). MRM Framework: Multi-Source Statistical Foundation
for Canadian Carceral, Police, and Oversight Data (v1). Zenodo.
https://doi.org/10.5281/zenodo.20096075

# Hawkes-process methodology paper
Ruhela, V. S. (2026). Criminological Hawkes Process via MORIE: Markovian
and Non-Markovian Self-Exciting Point Processes for Toronto Crime (v1).
Zenodo. https://doi.org/10.5281/zenodo.20102198

# Empirical applications paper
Ruhela, V. S. (2026). Solitary Confinement, Self-Excitation, and
Institutional Churn: Empirical Applications of MRM to Canadian Carceral
and Police Data (v1). Zenodo. https://doi.org/10.5281/zenodo.20175689
```

See [`CITATION.cff`](https://github.com/hadesllm/morie/blob/main/CITATION.cff) for machine-readable citation metadata.


## Acknowledgments

### AI assistance

MORIE was developed with substantial assistance from frontier AI
assistants. The author retains full responsibility for the code, the
methods, and the scientific claims; AI assistance accelerated
implementation but does not change the attribution of the work.

- **Claude — Anthropic.** Anthropic's Claude family (Opus, Sonnet, and
  Haiku across the 4.x generation) was used extensively throughout
  development for code generation, refactoring, documentation, code
  review, and design discussions. Use was supported by Anthropic
  research-credit programs.

- **Gemini and Vertex AI — Google.** Google's Gemini 2.5 models (Pro and
  Flash) on the Vertex AI platform were used extensively for additional
  code generation, cross-checking Claude-generated code, multi-modal
  data analysis, and prototype evaluation. Use was supported by Google
  research-credit programs.

### Funding and infrastructure

- Anthropic — Claude API research credits.
- Google — Gemini / Vertex AI research credits.
- The author thanks **Glenn McNamara** — a 35-year career with the
  Ontario Government — for his methodological mentorship. He brings
  distribution theory, applied-statistics intuition for administrative
  data, and the judgment that grounds much of this framework. Glenn
  is the **M** in **MRM (Multilevel Reconciliation Methodology; people-credit reading: McNamara-Ruhela-Medina)** (catalyst).

- The author thanks **Prof. Angela Zorro Medina**, Centre for
  Criminology and Sociolegal Studies, University of Toronto, who is
  the author's **supervisor**, **methodological instructor**, the
  **domain-expert reviewer** of the preliminary methodological
  approach, and a **knowledge user** of the framework. The
  methodological lineage MRM follows is established in her work on
  anti-gang legislation (Zorro Medina, 2023, *The Effect of
  Anti-Gang Laws on Crime and Social Control*) — staggered
  two-way-fixed-effects identification, formal leads-and-lags
  Granger-causality diagnostics for parallel trends, multi-source
  data-integration over five jurisdictional sources, deterrence /
  routine-activities / certainty mechanism categorisation, and the
  inequality-effects-of-criminal-law framing — all of which
  directly shape MRM's empirical-statistical spine. Prof. Medina is
  the **M** in MRM (supervisor & reviewer).

### Data acknowledgments

Several MRM analyses use Statistics Canada and Health Canada Public
Use Microdata Files (PUMFs) — including the **Canadian Cannabis
Survey (CCS)**, the **Canadian Student Alcohol and Drugs Survey
(CSADS)**, the **Canadian Substance Use Survey (CSUS)**, the
**Canadian Alcohol and Drugs Survey (CADS, 2019;
[doi.org/10.25318/132500052021001-eng](https://doi.org/10.25318/132500052021001-eng))**,
and the **Canadian Postsecondary Education Alcohol and Drug Use
Survey (CPADS)** — along with Public Health Agency of Canada (PHAC)
and Canadian Institute for Health Information (CIHI) aggregates.
Although the analyses use Statistics Canada and Health Canada data,
the analyses, interpretations, and conclusions are those of the
author and do not represent the views of Statistics Canada or
Health Canada. Ontario open data (OTIS, A01-RCDD release; via
`data.ontario.ca`) and Toronto Police Service open data are used
under the same standard disclaimer.

## License

MORIE adopts a **per-component licensing model** (revised in v0.7.0 to drop GPL-2.0 from the main R + Python distribution; only the optional Linux-kernel adjuncts remain GPL-2.0):

- **Python package** (`src/morie/`, `src/moirais/`) — dual-licensed `MIT OR Apache-2.0` (the Rust-ecosystem convention; recipient picks either).  See [`LICENSE-MIT`](https://github.com/hadesllm/morie/blob/main/LICENSE-MIT) and [`LICENSE-APACHE`](https://github.com/hadesllm/morie/blob/main/LICENSE-APACHE).
- **R package** (`r-package/morie/`, `r-package/moirais/`) — dual-licensed `Apache License (== 2) | MIT + file LICENSE` per CRAN convention; downstream R consumers pick either. See [`LICENSE-MIT`](https://github.com/hadesllm/morie/blob/main/LICENSE-MIT) and [`LICENSE-APACHE`](https://github.com/hadesllm/morie/blob/main/LICENSE-APACHE).
- **Optional Linux kernel adjuncts** (`kernel-module/morie.c`, `daemon/morie_lsm.py`) — `GPL-2.0-only` (Linux kernel ABI requires GPL for loaded modules; the userspace LSM-style audit daemon stays under the same license for symmetry). These are NOT part of the R / Python distribution; they are separately-licensed adjuncts. See [`kernel-module/LICENSE-GPL2`](https://github.com/hadesllm/morie/blob/main/kernel-module/LICENSE-GPL2).
- **Papers, data and documentation** — `CC-BY-4.0` unless explicitly marked otherwise.

The full per-component breakdown is documented in [`LICENSING.md`](https://github.com/hadesllm/morie/blob/main/LICENSING.md).

## Reporting issues / security

- General issues: [GitHub Issues](https://github.com/hadesllm/morie/issues)
- Security vulnerabilities: see [`SECURITY.md`](https://github.com/hadesllm/morie/blob/main/.github/SECURITY.md)
