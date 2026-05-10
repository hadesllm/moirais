# MOIRAIS

**Methods for Observational Inference and Robust Analysis of Interventions in Scientific Experimentation**

A multi-domain scientific computing toolkit (Python and R) for observational inference, with sociolegal, signal-processing, cryptographic, spatial-statistics, statistical-physics, and psychometrics modules. Hosts the DLRM framework as a flagship application for Canadian carceral, police, and oversight data analysis.

[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](LICENSE)
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

# Run a Ruhela formulation (DLRM flagship)
from moirais.otis_all_analyze import analyze_a01_ruhela_formulations
result = analyze_a01_ruhela_formulations(df)
print(result)
```

## Documentation

Full documentation is at [hadesllm.github.io/moirais](https://hadesllm.github.io/moirais/) (auto-built from `docs/source/`).

## Citation

If you use MOIRAIS in your research, please cite the package paper, the DLRM framework paper, and (where applicable to your work) the Hawkes-process methodology paper:

```
Ruhela, V. S. (2026). MOIRAIS: A Multi-Domain Scientific Computing
Toolkit for Observational Inference, with Sociolegal, Signal-Processing,
Cryptographic, and Spatial-Statistics Modules. Zenodo.
https://doi.org/10.5281/zenodo.20096350

Ruhela, V. S. (2026). The DLRM Framework: A Multi-Source Mathematical
Foundation for Canadian Carceral, Police, and Oversight Data, Implemented
as RF Modules in MOIRAIS. Zenodo.
https://doi.org/10.5281/zenodo.20096075

Ruhela, V. S. (2026). Criminological Hawkes Process via MOIRAIS:
Markovian and Non-Markovian Self-Exciting Point Processes for Toronto
Crime. Zenodo.
https://doi.org/10.5281/zenodo.20102198
```

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

## License

MOIRAIS is released under the GNU General Public License v2 (`GPL-2.0-only`); see [`LICENSE`](LICENSE). The licensing matrix for individual components is documented in [`LICENSING.md`](LICENSING.md).

## Reporting issues / security

- General issues: [GitHub Issues](https://github.com/hadesllm/moirais/issues)
- Security vulnerabilities: see [`SECURITY.md`](.github/SECURITY.md)
