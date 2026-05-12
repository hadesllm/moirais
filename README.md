# MORIE

**Multi-domain Open Research and Inferential Estimation**

> *Renamed from MOIRAIS in v0.1.3. The `moirais` Python module and the `moirais` R package remain available as deprecated aliases — `import morie` / `library(morie)` is the canonical entry point going forward.*

A multi-domain scientific computing toolkit (Python and R) for observational inference, with sociolegal, signal-processing, cryptographic, spatial-statistics, statistical-physics, and psychometrics modules. Hosts the MRM framework as a primary application for Canadian carceral, police, and oversight data analysis.

[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-d97706.svg)](https://github.com/hadesllm/morie/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/morie.svg)](https://pypi.org/project/morie/)
[![r-universe](https://img.shields.io/badge/r--universe-hadesllm-276DC3)](https://hadesllm.r-universe.dev/morie)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DOI · morie R](https://img.shields.io/badge/DOI%20%C2%B7%20morie%20R-10.5281%2Fzenodo.20111233-0d9488?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.20111233)
[![DOI · morie Python](https://img.shields.io/badge/DOI%20%C2%B7%20morie%20Python-10.5281%2Fzenodo.20096350-7c3aed?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.20096350)
[![MRM paper](https://img.shields.io/badge/MRM_paper-10.5281%2Fzenodo.20096075-15803d?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.20096075)
[![Hawkes paper](https://img.shields.io/badge/Hawkes_paper-10.5281%2Fzenodo.20102198-be123c?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.20102198)

## Installation

### Python (PyPI)

```bash
pip install morie
```

### R (CRAN)

```r
install.packages("morie")
```

### R (r-universe; nightly binary builds)

```r
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

## Documentation

Full documentation is at [hadesllm.github.io/morie](https://hadesllm.github.io/morie/).

## Citation

If you use morie in your research, please cite **both software papers** (R and Python)
and, where applicable, the **MRM framework paper** and the **Hawkes methodology paper**.

```
# Software paper — R (also the R package source on Zenodo)
Ruhela, V. S. (2026). morie: Multi-domain Open Research and Inferential
Estimation in R (v0.3.0). Zenodo.
https://doi.org/10.5281/zenodo.20111233

# Software paper — Python (also the Python package source on Zenodo)
Ruhela, V. S. (2026). morie: Multi-domain Open Research and Inferential
Estimation in Python (v0.3.0). Zenodo.
https://doi.org/10.5281/zenodo.20096350

# MRM framework paper (theoretical foundations)
Ruhela, V. S. (2026). MRM Framework: Multi-Source Statistical Foundation
for Canadian Carceral, Police, and Oversight Data (v1). Zenodo.
https://doi.org/10.5281/zenodo.20096075

# Hawkes-process methodology paper
Ruhela, V. S. (2026). Criminological Hawkes Process via MORIE: Markovian
and Non-Markovian Self-Exciting Point Processes for Toronto Crime (v1).
Zenodo. https://doi.org/10.5281/zenodo.20102198
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

MORIE adopts a **per-component licensing model** (since v0.3.0):

- **Python package** (`src/morie/`, `src/moirais/`) — dual-licensed `MIT OR Apache-2.0` (the Rust-ecosystem convention; recipient picks either).  See [`LICENSE-MIT`](https://github.com/hadesllm/morie/blob/main/LICENSE-MIT) and [`LICENSE-APACHE`](https://github.com/hadesllm/morie/blob/main/LICENSE-APACHE).
- **R package** (`r-package/morie/`, `r-package/moirais/`) — `GPL-2.0-only` (matches the R-ecosystem / CRAN convention).  See [`LICENSE-GPL2`](https://github.com/hadesllm/morie/blob/main/LICENSE-GPL2).
- **Linux kernel module** (`kernel-module/morie.c`) — `GPL-2.0-only` (kernel ABI requirement).
- **Papers, data and documentation** — `CC-BY-4.0` unless explicitly marked otherwise.

The full per-component breakdown is documented in [`LICENSING.md`](https://github.com/hadesllm/morie/blob/main/LICENSING.md).

## Reporting issues / security

- General issues: [GitHub Issues](https://github.com/hadesllm/morie/issues)
- Security vulnerabilities: see [`SECURITY.md`](https://github.com/hadesllm/morie/blob/main/.github/SECURITY.md)
