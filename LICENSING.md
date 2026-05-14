# Licensing

The `morie` project is dual-licensed under **MIT OR Apache-2.0** for
both language sides (Python and R), following the Rust-ecosystem
convention. The two optional Linux-kernel adjuncts remain
`GPL-2.0-only` because the kernel ABI requires it; they are *not*
part of the R or Python distribution.

> **Migration note (v0.7.0, 2026-05-14).** The R side was previously
> `GPL-2.0-only`. It is now dual-licensed `Apache License (== 2) | MIT`,
> matching the Python side. The historical analysis that led to the
> earlier choice is preserved in `LICENSING_ANALYSIS.md` with a
> "now-superseded" header. The rationale for the migration:
> GPL-2.0 is incompatible with Apache-2.0's patent clause, which made
> a dual-language Rust-style license infeasible; and several downstream
> distribution channels (Homebrew formulae, `curl ... | sh` installer
> trees, container layering) work more cleanly with permissive
> licensing.

## Python package (`src/morie/`, `src/moirais/`)

Licensed under **`MIT OR Apache-2.0`** — recipient picks either.

SPDX identifier: `MIT OR Apache-2.0`

License files:
- `LICENSE-MIT`
- `LICENSE-APACHE`

## R package (`r-package/morie/`, `r-package/moirais/`)

Licensed under **`Apache License (== 2) | MIT + file LICENSE`** —
the CRAN-form dual-license, recipient picks either.

CRAN DESCRIPTION field:
```
License: Apache License (== 2) | MIT + file LICENSE
```

License files:
- `LICENSE-MIT`
- `LICENSE-APACHE`
- `r-package/morie/LICENSE` and `r-package/moirais/LICENSE` (CRAN
  copyright-holder stubs declaring `YEAR: 2026 / COPYRIGHT HOLDER:
  Vansh Singh Ruhela`)

## Optional Linux-kernel adjuncts (`kernel-module/`, `daemon/`)

These two subtrees are **`GPL-2.0-only`** because:
- `kernel-module/morie.c` is a Linux kernel module. The Linux kernel
  ABI requires loaded modules to carry a GPL-compatible license, and
  GPL-2.0-only is the kernel's own license.
- `daemon/morie_lsm.py` is an LSM-style userspace audit daemon; it
  stays under the same license as the kernel module it pairs with
  for symmetry.

License file:
- `kernel-module/LICENSE-GPL2`

These subtrees are **not** distributed as part of the morie wheel
or the CRAN tarball. They are separately-licensed adjuncts that
users can opt into.

## Papers, data, documentation

The papers under `papers/`, the data references under
`data/datasets/userguides/`, and project-level documentation
(`README.md`, `docs/`) are released under **CC-BY-4.0** unless
explicitly marked otherwise.

## Why dual-license MIT + Apache-2.0?

The Rust-ecosystem standard of `MIT OR Apache-2.0` ensures:
- **MIT** for projects that want minimal license obligations;
- **Apache-2.0** for projects that need the explicit patent grant.

Either license is sufficient on its own; recipients pick whichever
suits their downstream needs. Applying this same pattern to the R
side keeps the two language packages license-compatible and avoids
the GPL-2.0 / Apache-2.0 patent-clause conflict that the earlier
mixed-license model produced.
