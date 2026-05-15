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
| [`morie`](https://github.com/hadesllm/morie) | Python + R + Linux | Multi-domain Open Research and Inferential Estimation | AGPL-3.0-or-later (kernel adjuncts: GPL-2.0-only) |
| [`clew`](https://github.com/hadesllm/clew) | Rust (Zig planned) | Git plumbing with packfile + SSH remotes | AGPL-3.0-or-later |
| [`MOIRAIS`](https://github.com/hadesllm/morie) | R + Python alias | Deprecated alias for `morie` (v0.1.x compatibility) | matches `morie` |
| [`keyserver`](https://github.com/hadesllm/keyserver) | Python (private) | Per-user API key issuance + audit | All Rights Reserved |

## Licensing

Licences vary per project — see the table above and each repository's
`LICENSING.md`. In brief:

- **`morie`** — `AGPL-3.0-or-later` (Python and R). Optional Linux
  kernel adjuncts are `GPL-2.0-only` (kernel ABI requirement). Papers,
  data, and documentation are `CC BY-NC-SA 4.0`.
- **`clew`** — `AGPL-3.0-or-later`.
- **`keyserver`** — All Rights Reserved (private).

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
- **Licensing matters.**  Contributions are accepted under the licence
  of the project being contributed to — see that project's
  `LICENSING.md` and `CONTRIBUTING.md`.
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
