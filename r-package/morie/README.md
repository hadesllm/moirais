# morie

`morie` is a dual-language (Python + R) scientific computing package for epidemiological and statistical modeling.

## What it does

- **87 exported R functions** across causal inference, sampling, psychometrics, OTIS correctional analysis, and more.
- Mirrors the Python `morie.fn` library (5724 individual functions across 218 categories) for cross-language parity.
- Reads and validates `outputs_manifest.csv` tables.
- Audits whether declared public artifacts are present on disk.
- Builds output manifests from a directory of generated files.
- Runs project workflow steps (`modules`, `publish`, `render`, `readiness`) from R.
- Provides CPADS contract helpers and IPW/eBAC workflow functions.
- Accesses the shared SQLite database (`morie_datasets.db`) with 41 built-in datasets.
- Generates synthetic epidemiology-style tabular data for development/testing.
- Provides an optional assistant bridge to the Python LLM integration.

## Scientific guardrail

- Synthetic data should be used for development, testing, demos, and CI only.
- Final inferential or policy-facing results must be produced from approved real data with full provenance.
- Synthetic runs should be explicitly labeled as synthetic in outputs and reporting text.

## Install from local source

```r
install.packages("r-package/morie", repos = NULL, type = "source")
```

The assistant bridge now supports local fallback mode through the Python
package when no live OpenAI credentials are configured.

## Example

```r
library(morie)

manifest <- read_outputs_manifest(project_root = "/path/to/project")
audit <- audit_public_outputs(project_root = "/path/to/project", manifest = manifest)
summary <- summarize_output_audit(audit)

summary
```

## Synthetic data example

```r
library(morie)

synthetic_path <- write_synthetic_data(
  path = "data/private/synthetic_study_data.csv",
  n = 8000,
  seed = 2026,
  overwrite = TRUE
)

synthetic_path
```

## Cross-project adaptation

```r
library(morie)

name_map <- default_synthetic_name_map("generic")
name_map["cannabis_use"] <- "exposure_any"
name_map["bac"] <- "outcome_continuous"

dat <- generate_synthetic_data(
  n = 5000,
  seed = 1,
  name_map = name_map
)
```

## Citation

Use `citation("morie")` after installation. Please cite **both**
the software and the companion paper.

```bibtex
@Manual{ruhela_morie_R_2026,
  title   = {morie: Multi-domain Open Research and Inferential Estimation in R},
  author  = {Ruhela, Vansh Singh},
  year    = {2026},
  note    = {R package version 0.9.4},
  doi     = {10.5281/zenodo.20111233},
  url     = {https://github.com/hadesllm/morie}
}

@Misc{ruhela_morie_python_2026,
  title     = {morie: Multi-domain Open Research and Inferential Estimation in Python},
  author    = {Ruhela, Vansh Singh},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20096350},
  url       = {https://doi.org/10.5281/zenodo.20096350}
}

@Misc{ruhela_mrm_framework_2026,
  title     = {MRM: Multilevel Reconciliation Methodology --- A multi-source statistical foundation for Canadian carceral, police, and oversight data},
  author    = {Ruhela, Vansh Singh},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20096075},
  url       = {https://doi.org/10.5281/zenodo.20096075}
}

@Misc{ruhela_hawkes_2026,
  title     = {Criminological Hawkes Process via MORIE: Markovian and Non-Markovian Self-Exciting Point Processes for Toronto Crime},
  author    = {Ruhela, Vansh Singh},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20102198},
  url       = {https://doi.org/10.5281/zenodo.20102198}
}

@Misc{ruhela_empirical_2026,
  title     = {Solitary Confinement, Self-Excitation, and Institutional Churn: Empirical Applications of MRM to Canadian Carceral and Police Data},
  author    = {Ruhela, Vansh Singh},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20175689},
  url       = {https://doi.org/10.5281/zenodo.20175689}
}
```
