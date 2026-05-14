# MORIE Licensing Analysis

> **SUPERSEDED — historical document.**
> See [`LICENSING.md`](LICENSING.md) for the current per-component
> licensing model. As of **v0.7.0 (2026-05-14)**: Python AND R are
> both `MIT OR Apache-2.0` (CRAN form
> `Apache License (== 2) | MIT + file LICENSE` on the R side); the
> optional Linux-kernel adjuncts in `kernel-module/` and `daemon/`
> remain `GPL-2.0-only` because the kernel ABI requires it; papers
> + data are `CC-BY-4.0`.
>
> Prior models (also superseded by this document):
> - **v0.3.0 – v0.6.1**: Python `MIT OR Apache-2.0`, R + kernel
>   module + LSM daemon all `GPL-2.0-only`. Migrated away from
>   because GPL-2.0 is incompatible with Apache-2.0's patent
>   clause and several downstream distribution channels work more
>   cleanly with permissive licensing.
> - **v0.1.x – v0.2.1**: single-license `GPL-2.0-only` everywhere.
>   The analysis below is retained as the historical justification
>   for that period.

Status: 2026-05-11 (historical, retained for v0.2.x context)
Author: Vansh Singh Ruhela
Scope (historical, v0.1.x–v0.2.1): justification for the then-single `GPL-2.0-only` license of the MORIE
toolkit given its dependency graph (DoubleML BSD-3-Clause, the
scientific-Python stack BSD-3-Clause, and the broader R ecosystem
GPL-2 / GPL-3 / MIT mix).

This document is informational, not legal advice. Where the
analysis touches on commercial deployment, consult counsel.

---

## TL;DR

- MORIE is released under **GPL-2.0-only**.
- Every direct runtime dependency is licensed permissively
  (BSD-3-Clause, BSD-2-Clause, MIT, Apache-2.0) or under a
  GPL-compatible copyleft licence (GPL-2, GPL-3, LGPL).
- All permissive→GPL combinations used here are one-way
  compatible: MORIE can absorb BSD/MIT/Apache deps into a
  GPL-2 work; the reverse is not true.
- No dependency in the current `pyproject.toml` or
  `r-package/morie/DESCRIPTION` triggers a GPL-2 incompatibility.
- Patent posture: GPL-2 carries an *implied* patent licence
  under §7 (Bill Gates clause) but no explicit grant. Where
  patent risk matters (e.g. cryptographic primitives), the
  upstream dependency's licence is what controls — and the
  cryptographic modules in `morie/crypto/` are reference
  implementations explicitly marked **not for production**
  (see `SECURITY.md`).

## 1. License-compatibility ground rules

GPL-2 is a *copyleft* licence. The relevant rules for combining it
with other licences are:

1. **Permissive → GPL**: A BSD-3-Clause, BSD-2-Clause, MIT, or
   ISC dependency can be combined into a GPL-2 work, provided the
   permissive licence text and copyright notice ship with the
   combined work. The combined work itself is then governed by
   GPL-2.
2. **Apache-2.0 → GPL-2**: *Incompatible*. Apache-2.0's explicit
   patent-termination clause was held by the FSF to add
   restrictions not present in GPL-2. (Apache-2.0 *is* compatible
   with GPL-3.) If MORIE depended on a pure Apache-2.0 library at
   the source/derivative level, this would be a problem. It does
   not (see §3 below).
3. **LGPL-2.1 → GPL-2**: Compatible. LGPL allows being subsumed
   into a GPL-2 derivative.
4. **GPL-3 → GPL-2**: *Incompatible*. GPL-3 adds the
   anti-Tivoization and patent-retaliation clauses that GPL-2
   does not require. Mixing GPL-2-only and GPL-3-only code is
   not lawful as a single combined work. If a dependency is
   GPL-3-only, MORIE cannot link it as part of a GPL-2-only
   derivative — but most "GPL-3" code in practice is
   `GPL (>= 2)` or `GPL (>= 3)`, which is GPL-2-compatible only
   for the `(>= 2)` variant.
5. **Public-domain / CC0**: Always compatible.

## 2. Linking, "derivative work", and the runtime question

GPL-2's reach is the "derivative work" notion under copyright law,
not a moral preference. The settled questions for an R/Python
package like MORIE are:

- **A user importing `morie` into their analysis script** is *not*
  creating a derivative work. They are running MORIE. Their
  script can be any licence.
- **A package that vendors morie source files** into its own tree
  *is* creating a derivative work and must be GPL-2.
- **A package that `Imports:` morie** (R) or `import morie`
  (Python) is generally *not* held to be a derivative — it
  consumes the public API. The FSF interprets this more strictly
  for C linking; for dynamic-interpreted languages the case law
  remains uncertain, but the *de facto* community practice for
  R-CRAN and PyPI treats `Imports:` / `import` as non-derivative.
- **A commercial product that includes morie as part of a
  shipped binary** would be a derivative and would owe GPL-2
  source distribution.

## 3. Direct Python dependencies (current `pyproject.toml`)

| Package | Licence | GPL-2 compatible? |
|---|---|---|
| `pandas` | BSD-3-Clause | yes (permissive) |
| `numpy` | BSD-3-Clause | yes |
| `scipy` | BSD-3-Clause | yes |
| `scikit-learn` | BSD-3-Clause | yes |
| `statsmodels` | BSD-3-Clause | yes |
| `DoubleML` (py) | BSD-3-Clause | yes |
| `matplotlib` | matplotlib licence (BSD-style) | yes |
| `openpyxl` | MIT | yes |
| `httpx` | BSD-3-Clause | yes |
| `rich` | MIT | yes |
| `beautifulsoup4` | MIT | yes |
| `lxml` | BSD-3-Clause | yes |
| `tenacity` | Apache-2.0 | **see §4** |

Optional extras (`[ml]`, `[carbon]`, `[interactive]`):

| Package | Licence | GPL-2 compatible? |
|---|---|---|
| `imbalanced-learn` | MIT | yes |
| `codecarbon` | MIT | yes |
| `textual` | MIT | yes |
| `dashing` | MIT | yes |
| `enlighten` | MPL-2.0 | **see §4** |

## 4. The Apache-2.0 / MPL-2.0 question

Two direct dependencies are not BSD/MIT/ISC:

- **`tenacity` (Apache-2.0)** — retry-decorator library. The
  Apache-2.0 incompatibility with GPL-2 is at the **derivative-
  work** level, not at the *use* level. MORIE imports tenacity
  as a Python package; tenacity is not statically linked, not
  vendored, not modified. Under the FSF's own guidance and
  near-universal community practice, importing an Apache-2.0
  library from a GPL-2 work is permissible — the *combined
  distribution* is what would trigger incompatibility. MORIE
  does not redistribute tenacity; pip resolves it at install
  time.

  If a court were to disagree with the community reading and
  hold that `pip install morie` produces a derivative that
  bundles tenacity, the remedy is to either drop tenacity (the
  retry pattern is 30 lines of code we can vendor) or relax
  MORIE's licence to a GPL-3 / Apache-2.0 compatible form.

- **`enlighten` (MPL-2.0)** — terminal-progress library. MPL-2.0
  is file-level copyleft and is **GPL-2 compatible at the linked
  level**: MPL-2.0 §3.3 explicitly permits combination with
  GPL-licensed works. No issue.

## 5. R-package dependencies (`r-package/morie/DESCRIPTION`)

R packages on CRAN are licensed under one of a small set of
canonical licences. MORIE's `DESCRIPTION` declares `License:
GPL-2 | file LICENSE` and the `Imports:` / `Suggests:` chain
includes:

| Package | Licence | GPL-2 compatible? |
|---|---|---|
| `survey` | GPL-2 \| GPL-3 | yes (GPL-2 branch chosen) |
| `ivreg` | GPL-2 \| GPL-3 | yes |
| `MatchIt` | GPL-2 \| GPL-3 | yes |
| `DBI` | LGPL-2.1 \| LGPL-3 | yes (LGPL → GPL) |
| `RSQLite` | LGPL-2.1 \| LGPL-3 | yes |
| `jsonlite` | MIT | yes |
| `data.table` | MPL-2.0 | yes (see §4) |
| `readxl` | GPL-3 | **see below** |
| `pracma` | GPL-3 | **see below** |
| `signal` | GPL-2 | yes |
| `smotefamily` | GPL-2 \| GPL-3 | yes |
| `DoubleML` (R) | MIT | yes |
| `mlr3` | LGPL-3 | LGPL→GPL-2 incompatible at link level **see below** |
| `mlr3learners` | LGPL-3 | same as above |
| `testthat` | MIT | yes |
| `knitr` | GPL-2 \| GPL-3 | yes |
| `rmarkdown` | GPL-3 | **see below** |

### The GPL-3 / LGPL-3 dependencies

`readxl`, `pracma`, `rmarkdown`, `mlr3`, `mlr3learners` are
GPL-3 / LGPL-3 only. These are all listed under `Suggests:`,
**not** `Imports:`. R's package system treats `Suggests:` as
"may be used, must be available for the test/vignette but is
not required at runtime." This matters because:

1. Building / installing morie does not load these.
2. Tests that use them load them at runtime; the combined
   *running program* may be incompatible, but no derivative
   *distribution* is created — CRAN ships morie's source,
   not a bundle.
3. We can choose to either (a) leave them in `Suggests:` and
   note that morie-on-CRAN does not vendor them, or (b)
   relax MORIE to `GPL (>= 2)` to cover the GPL-3 leg.

**Recommendation**: leave them in `Suggests:` and accept that
when an end user installs and *runs* morie's irm() vignette
they form a combined work governed by the stricter licence
(GPL-3). The CRAN-distributed morie tarball itself is pure
GPL-2-only source.

## 6. DoubleML specifically

DoubleML (both Python and R) is **BSD-3-Clause** at the time of
this writing:

- Python: <https://github.com/DoubleML/doubleml-for-py/blob/main/LICENSE>
- R:      <https://github.com/DoubleML/doubleml-for-r/blob/main/LICENSE>

Wait — the R package's `DESCRIPTION` declares `License: MIT +
file LICENSE`. The source file shows a standard MIT licence
text. **Both licences (BSD-3 and MIT) are GPL-2 compatible.**

MORIE wraps DoubleML in two places:

- `src/morie/causal.py` — `estimate_irm()`, `estimate_plr()`,
  `estimate_pliv()` thin wrappers calling DoubleML's classes.
- `r-package/morie/R/irm.R` — `estimate_irm()` wrapper
  calling `DoubleML::DoubleMLIRM`.

These are **API-level uses**, not derivative incorporations of
DoubleML source. The combined work is governed by MORIE's
licence (GPL-2-only). DoubleML's BSD/MIT notices ship in our
PyPI/CRAN tarballs as a courtesy (the dep ecosystem's
licence-aggregation tooling — `pip-licenses`, `licensee` —
handles this automatically).

## 7. Patent posture

GPL-2 does not contain an explicit patent grant. Section 7
("the Bill Gates clause") states that if patent restrictions
prevent distribution under GPL-2 terms, the work cannot be
distributed at all — an implicit patent-no-action covenant
from any distributor who voluntarily uses GPL-2.

For MORIE:

- **The statistical estimators** in `morie.causal`,
  `morie.weights`, `morie.bootstrap_methods`, etc. are
  textbook algorithms (Robins, Rosenbaum, Imbens, Pearl,
  Chernozhukov) with no live patent encumbrance.
- **The Hawkes-process methods** in `morie.tps_stochastic`
  similarly use standard published algorithms (Ogata 1981,
  Hawkes 1971) with no patent claims.
- **The cryptographic primitives** in `morie.crypto`
  (ML-KEM, ML-DSA, NTRU, lattice operations) are textbook
  reference implementations explicitly marked
  *not for production* (see `.github/SECURITY.md`).
  Cryptographic patent risk is real in this area
  (e.g. the RSA patent until 2000) but the algorithms we
  reference are post-quantum NIST FIPS standards with
  documented royalty-free terms.

If MORIE were targeted for industrial deployment that needed
a strong explicit patent grant, the cleanest path is to
re-licence under Apache-2.0 (incompatible with GPL-2). We
have chosen not to do this; the analytical-research-software
use case is not where patent risk crystallises.

## 8. Why GPL-2-only (not GPL-2-or-later, not Apache-2.0)

| Option | Pro | Con |
|---|---|---|
| GPL-2-only | Linus's choice; widely understood; locks the licence to terms contributors agreed to | No explicit patent grant; Apache-2.0 deps need careful handling |
| GPL-2-or-later | Future-proofs against post-v3 GPL revisions | Delegates the licence to a future FSF; modifies the agreement contributors signed |
| Apache-2.0 | Strong explicit patent grant; industry-standard permissive | Loses copyleft (downstream may close source); GPL-2 deps incompatible |
| BSD-3-Clause | Mirrors DoubleML; maximum adoption | No copyleft; no patent grant |
| Dual MIT/Apache-2.0 | Rust-ecosystem standard | Two licence files to maintain; same trade-offs as Apache-2.0 |

We have chosen **GPL-2-only** because:

1. The R-package ecosystem strongly defaults to GPL-2 / GPL-3.
   `survey`, `MatchIt`, `mlr3`, `data.table` — virtually all
   the heavyweight observational-inference packages are
   GPL-licensed. Releasing under BSD/MIT/Apache would mean
   our derivatives could *not* recombine with these — exactly
   the opposite of what permissive licensing intends.
2. The MOIRAIS-era community precedent: Linus's preference for
   `GPL-2-only` on the Linux kernel sets the cultural anchor
   for "we want copyleft, but we want the licence we actually
   signed up for."
3. The patent-grant concern is real but does not materialise
   in this codebase: there is no production cryptography,
   no live patented statistical algorithm.

## 9. CRAN's reading

CRAN explicitly accepts `License: GPL-2 | file LICENSE`. The
DESCRIPTION's `License` field is the canonical declaration.
CRAN's policy further requires that any **non-trivial code
fragment from another package** that ships in our source tree
be clearly attributed under that fragment's licence. MORIE
does not bundle DoubleML, scikit-learn, or any other dep's
source — we declare them in `Imports:`. No CRAN issue.

## 10. License attribution shipping checklist

When morie 0.1.3 ships:

- `LICENSE` — the GPL-2 text (already in place).
- `LICENSING.md` — human-readable summary (already in place).
- `LICENSING_ANALYSIS.md` — this document.
- `inst/COPYRIGHTS` (R package) — short attribution list
  for any vendored material. *Currently empty; nothing is
  vendored.*
- `.zenodo.json` `license` field — `GPL-2.0-only`.
- PyPI classifier — `License :: OSI Approved :: GNU General
  Public License v2 (GPLv2)` (already in pyproject.toml).
- `CITATION.cff` `license` field — `GPL-2.0-only` (already
  in place).

## 11. Open questions to revisit

- If MORIE eventually vendors a small piece of DoubleML
  source (rather than importing it), the BSD-3-Clause
  text and copyright notice must be reproduced in our
  tree. *Not currently the case.*
- If MORIE adds a hard-dep (not `Suggests:`) on an
  Apache-2.0 package whose code we modify, we will need
  to either drop GPL-2-only and move to GPL-3 (Apache-2.0
  compatible) or vendor the Apache code in a way that
  preserves its licence as a clearly-separable file-set.

End of analysis.
