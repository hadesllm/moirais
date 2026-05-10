# MOIRAIS

**Methods for Observational Inference and Robust Analysis of Interventions in Scientific Experimentation**

A multi-domain scientific computing toolkit (Python and R) for observational inference, with sociolegal, signal-processing, cryptographic, spatial-statistics, statistical-physics, and psychometrics modules. Hosts the MRM framework as a primary application for Canadian carceral, police, and oversight data analysis.

[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](https://github.com/hadesllm/moirais/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/moirais.svg)](https://pypi.org/project/moirais/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20096350-blue)](https://doi.org/10.5281/zenodo.20096350)

## Installation

### Python (PyPI)

```bash
pip install moirais
```

### R (CRAN)

```r
install.packages("moirais")
```

### R (r-universe; nightly binary builds)

```r
install.packages(
  "moirais",
  repos = c(
    hadesllm = "https://hadesllm.r-universe.dev",
    CRAN     = "https://cloud.r-project.org"
  )
)
```

## Quick start

```python
import moirais

# Load a built-in dataset
df = moirais.load_dataset("otis-2025")

# Run an MRM module on OTIS data
from moirais.otis_all_analyze import analyze_a01_mrm
result = analyze_a01_mrm(df)
print(result)
```

## Documentation

Full documentation is at [hadesllm.github.io/moirais](https://hadesllm.github.io/moirais/).

## Citation

If you use MOIRAIS in your research, please cite the package paper, the MRM framework paper, and (where applicable to your work) the Hawkes-process methodology paper:

```
Ruhela, V. S. (2026). MOIRAIS: A Multi-Domain Scientific Computing
Toolkit for Observational Inference, with Sociolegal, Signal-Processing,
Cryptographic, and Spatial-Statistics Modules. Zenodo.
https://doi.org/10.5281/zenodo.20096350

Ruhela, V. S. (2026). The MRM Framework: A Multi-Source Statistical
Foundation for Canadian Carceral, Police, and Oversight Data, Implemented
as MRM Modules in MOIRAIS. Zenodo.
https://doi.org/10.5281/zenodo.20096075

Ruhela, V. S. (2026). Criminological Hawkes Process via MOIRAIS:
Markovian and Non-Markovian Self-Exciting Point Processes for Toronto
Crime. Zenodo.
https://doi.org/10.5281/zenodo.20102198
```

See [`CITATION.cff`](https://github.com/hadesllm/moirais/blob/main/CITATION.cff) for machine-readable citation metadata.

## Acknowledgments

### AI assistance

MOIRAIS was developed with substantial assistance from frontier AI
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
  is the **M** in **MRM (McNamara-Ruhela-Medina)** (catalyst).

- The author thanks **Prof. Angela Zorro Medina**, Centre for
  Criminology and Sociolegal Studies, University of Toronto, for
  expert review of the framework and for the methodological lineage
  established by her work on anti-gang legislation (Zorro Medina,
  2023, *The Effect of Anti-Gang Laws on Crime and Social
  Control*) — staggered two-way-fixed-effects identification,
  formal leads-and-lags Granger-causality diagnostics for parallel
  trends, multi-source data-integration over five jurisdictional
  sources, deterrence / routine-activities / certainty mechanism
  categorisation, and the inequality-effects-of-criminal-law
  framing — all of which directly shape MRM's empirical-statistical
  spine. Prof. Medina is the **M** in MRM (reviewer).

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

MOIRAIS is released under the GNU General Public License v2 (`GPL-2.0-only`); see [`LICENSE`](https://github.com/hadesllm/moirais/blob/main/LICENSE). The licensing matrix for individual components is documented in [`LICENSING.md`](https://github.com/hadesllm/moirais/blob/main/LICENSING.md).

## Reporting issues / security

- General issues: [GitHub Issues](https://github.com/hadesllm/moirais/issues)
- Security vulnerabilities: see [`SECURITY.md`](https://github.com/hadesllm/moirais/blob/main/.github/SECURITY.md)
