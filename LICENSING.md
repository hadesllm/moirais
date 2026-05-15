# Licensing

The `morie` project is licensed under the **GNU Affero General Public
License, version 3.0 or later (`AGPL-3.0-or-later`)** for both language
sides (Python and R). The move to a strong copyleft licence is
deliberate: morie supports research into carceral, police, and
oversight accountability, and the AGPL guarantees that any modified
version distributed to others — or offered to users over a network —
must publish its source. Improvements to the work cannot be taken
private; the work, and any changes to it, stay visible to the public.

## Why AGPL-3.0

- **No closed forks.** Any conveyed modified version of morie must
  itself be licensed `AGPL-3.0-or-later`, with complete source
  (AGPL §5).
- **No hidden network / SaaS modifications.** A modified morie offered
  to users over a network must provide those users its Corresponding
  Source (AGPL §13) — the clause that distinguishes the AGPL from the
  ordinary GPL.
- **Attribution is preserved.** Copyright and licence notices must be
  kept in all copies and modified versions; removing them is a licence
  violation.

morie remains genuine, OSI-approved open-source software. The AGPL does
not restrict who may use morie or for what purpose; it ensures that
modifications and improvements cannot be kept secret.

## Python package (`src/morie/`)

Licensed under **`AGPL-3.0-or-later`**.

SPDX identifier: `AGPL-3.0-or-later`

Licence file: `LICENSE` (the full AGPL-3.0 text).

## R package (`r-package/morie/`)

Licensed under **`AGPL-3`** — the R `DESCRIPTION` form:

```
License: AGPL-3
```

## Optional Linux-kernel adjuncts (`kernel-module/`, `daemon/`)

These two subtrees remain **`GPL-2.0-only`** because:

- `kernel-module/morie.c` is a Linux kernel module; the Linux kernel
  ABI requires loaded modules to carry the kernel's own `GPL-2.0-only`
  licence.
- `daemon/morie_lsm.py` is an LSM-style userspace audit daemon; it
  stays under the same licence as the kernel module it pairs with.

Licence file: `kernel-module/LICENSE-GPL2`.

These subtrees are **not** distributed as part of the morie wheel or
the R package. They are separately-licensed, independently-distributed
adjuncts that users can opt into; they are not combined with the
AGPL-licensed package.

## Papers, data, documentation

The papers under `papers/`, the data references under
`data/datasets/userguides/`, and project-level documentation
(`README.md`, `docs/`) are released under **CC BY-NC-SA 4.0**
(Creative Commons Attribution-NonCommercial-ShareAlike 4.0) unless
explicitly marked otherwise. NonCommercial: these materials may not be
used for commercial purposes without permission. ShareAlike:
adaptations must carry the same license.
