<!--
This file is intended for the `hadesllm/.github` repository — placed
at `profile/README.md` — to appear on the GitHub org landing page
at https://github.com/hadesllm.

To deploy:
  git clone git@github.com:hadesllm/.github.git
  mkdir -p .github/profile
  cp /path/to/morie/docs/hadesllm-org-profile/README.md profile/README.md
  git add profile/README.md
  git commit -m "feat: org profile README v0.3.0"
  git push origin main
-->

# hadesllm

Open-source scientific computing for **causal inference, observational
data, signal processing, and Canadian carceral / police oversight
research**.  Packages here are designed to be usable across the
Python + R ecosystem with no lockjaw — full dependency bootstrap,
clear per-component licensing, and reproducible papers backed by
Zenodo DOIs.

## Active packages

| Package | Language | Purpose | License |
| --- | --- | --- | --- |
| [`morie`](https://github.com/hadesllm/morie) | Python + R + Linux | Multi-domain Open Research and Inferential Estimation | Py: MIT OR Apache-2.0 / R + kernel: GPL-2.0-only |
| [`clew`](https://github.com/hadesllm/clew) | Rust (Zig planned) | Git plumbing with packfile + SSH remotes | MIT OR Apache-2.0 |
| [`MOIRAIS`](https://github.com/hadesllm/morie) | R + Python alias | Deprecated alias for `morie` (v0.1.x compatibility) | matches `morie` |
| [`keyserver`](https://github.com/hadesllm/keyserver) | Python (private) | Per-user API key issuance + audit | All Rights Reserved |

## Licensing

`hadesllm` adopts a **per-component licensing model** (since 2026-05-12):

- **Python surfaces** (`pip install <pkg>`) are dual-licensed
  `MIT OR Apache-2.0` — the Rust-ecosystem convention; recipient picks either.
- **R surfaces** (CRAN + r-universe) are `GPL-2.0-only` — matches the
  R-ecosystem / CRAN convention.
- **Linux kernel modules** (where present) are `GPL-2.0-only` — kernel
  ABI requirement.
- **Papers, data, and documentation** are `CC-BY-4.0` unless explicitly
  marked otherwise.

The full per-package breakdown is in each package's `LICENSING.md`.

## Security disclosure

Vulnerability reports: see `SECURITY.md` (this org repo) or the
per-package `SECURITY.md`.  Encrypted: <hadesllm@proton.me>
(ProtonMail end-to-end).  Public PGP key is published in each
package's `SECURITY.md` and at <https://keyserver.hadesllm.com> (when
the keyserver is public).

## Contributing

Each package carries its own `CONTRIBUTING.md`.  The shared posture:

- **CI is the gate, not human gatekeeping.**  If the build passes, the
  contribution is welcome.  Maintainer review confirms scope fit, not
  correctness (CI already proved that).
- **Per-component licensing matters.**  Contributors to the Python side
  agree to dual `MIT OR Apache-2.0`; contributors to the R side agree
  to `GPL-2.0-only`.  CLA-free; the contribution itself carries the
  intent (per the *Developer Certificate of Origin*).
- **No lockjaw with morie.**  Dependency bootstrap is a feature.  Any
  user-facing path should detect missing Python/R/system tools and
  offer to install them, not fail with "dependency not found."

## Acknowledgements

The hadesllm ecosystem builds on extensive prior open-source work:
DoubleML (Chernozhukov et al.), spdep, gstat, lifelines, scikit-learn,
statsmodels, R-core, the Python Packaging Authority, r-universe, JOSS,
and the Bayesian-nonparametrics, empirical-process, and signal-processing
research communities.  Specific paper-by-paper attribution is in each
package's bibliography.
