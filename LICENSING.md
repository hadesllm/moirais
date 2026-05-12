# Licensing

The `morie` project is dual-licensed by language component, following
the Rust-ecosystem convention for the Python side and the R-ecosystem
convention for the R side.

## Python package (`src/morie/`, `src/moirais/`)

Licensed under **MIT OR Apache-2.0** — recipient picks either at their
option.  This is the standard Rust crate dual-license pattern, chosen
here to maximise downstream-adoption freedom and to keep `morie` usable
in any Python project regardless of that project's own licensing.

SPDX identifier: `MIT OR Apache-2.0`

License files:
- `LICENSE-MIT`
- `LICENSE-APACHE`

## R package (`r-package/morie/`, `r-package/moirais/`)

Licensed under **GPL-2.0-only**, matching the R-ecosystem convention
and CRAN's preferred default.  Both the canonical `morie` R package
and the deprecated `moirais` alias are GPL-2.0-only.

SPDX identifier: `GPL-2.0-only`

License file:
- `LICENSE-GPL2`

## Papers, data, documentation

The papers under `papers/`, the data references under
`data/datasets/userguides/`, and project-level documentation
(`README.md`, `docs/`) are released under **CC-BY-4.0** unless
explicitly marked otherwise.

## Why dual-license the Python side?

The Rust-ecosystem standard of `MIT OR Apache-2.0` ensures:
- **MIT** for projects that want minimal license obligations;
- **Apache-2.0** for projects that need the explicit patent grant.

Either license is sufficient on its own; recipients pick whichever
suits their downstream needs.
