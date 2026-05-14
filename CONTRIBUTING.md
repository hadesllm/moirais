# Contributing to morie

Thanks for your interest in contributing! morie is a dual-language
(Python + R) scientific computing toolkit; contributions of all sizes
are welcome.

This document follows the Rust-ecosystem contribution model: short,
permissive, and standard GitHub PR workflow.

## License of contributions

By submitting a contribution you agree that it is dual-licensed under
**MIT OR Apache-2.0** at the recipient's option (the same dual-license
that covers both the Python and R packages).

Contributions to the `kernel-module/` and `daemon/` subtrees are
licensed under **GPL-2.0-only** (Linux kernel ABI requirement); flag
in your PR if you are touching those subtrees.

No CLA, no DCO sign-off required. Just opening the PR is the
agreement.

## How to contribute

1. **Open an issue first** if your change is non-trivial — it saves
   re-work later. For tiny fixes (typos, doc improvements, obvious
   bugs) feel free to skip straight to a PR.
2. **Fork + branch + PR.** Standard GitHub flow.
3. **Test locally** before opening the PR:
   - Python: `pytest -q tests/`
   - R: `R CMD check --as-cran r-package/morie`
   - If you touched the LaTeX papers under `papers/`, all five rebuild
     with `xelatex main.tex && bibtex main && xelatex main.tex &&
     xelatex main.tex`.
4. **Update docs** if your change is user-visible:
   - Python API change: docstring + `tests/`.
   - R API change: roxygen2 docstring + regenerate `.Rd` with
     `devtools::document()` + commit the regenerated files.
   - User-facing behaviour: `r-package/morie/NEWS.md` and/or
     `ROADMAP.md`.

## Commit messages

Keep them readable. Imperative mood ("add X", "fix Y", "refactor Z").
Wrap the body at ~72 chars. Explain *why* the change is needed when
the diff itself doesn't make it obvious.

If your PR is logically more than one change, split it into multiple
commits — each one should compile and pass tests on its own.

## Coding style

- **Python**: PEP 8 with `ruff format` (black-compatible). Type hints
  on public-API functions. Numpydoc-style docstrings.
- **R**: 2-space indentation, snake_case, roxygen2 docstrings on every
  exported function. Every R file under `r-package/morie/R/` carries
  an SPDX header `# SPDX-License-Identifier: MIT OR Apache-2.0`.
- **LaTeX (papers)**: JSS document class; long filenames in
  `\footnote{...}` so the prose can break cleanly.

## Code of conduct

Please be respectful. We follow the
[Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/);
the full text is in [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
Reports go to <hadesllm@proton.me>.

## Security disclosures

**Do not file security issues in the public tracker.**
See [`.github/SECURITY.md`](.github/SECURITY.md) for the disclosure
address and embargo policy.

## Maintainer

Vansh Singh Ruhela
([@rootcoder007](https://github.com/rootcoder007),
<hadesllm@proton.me>,
ORCID [0009-0004-1750-3592](https://orcid.org/0009-0004-1750-3592)).

## A note on the dual MIT + Apache-2.0 license

The Rust-ecosystem convention is to dual-license under MIT OR
Apache-2.0 so that:

- **MIT** suits projects that want minimal license obligations;
- **Apache-2.0** suits projects that need the explicit patent grant.

Either license is sufficient on its own — recipients pick whichever
suits their downstream needs. See [`LICENSING.md`](LICENSING.md) for
the per-component breakdown.
