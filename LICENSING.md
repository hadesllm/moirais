# Licensing

This repo (and its sibling `wlp/`) contains code under several licenses,
chosen per-component to reflect what each piece is for and who should
be able to use it.  This document is the authoritative map.

## TL;DR

| Component | Path(s) | License | Why |
|--|--|--|--|
| **clew** (the VCS) | `dev/clew/` | **MIT OR Apache-2.0** | Designed for adoption — anyone (Anthropic, Google, Microsoft, anyone with a corporate legal team) can use without legal review.  Standard Rust ecosystem dual-license. |
| **MOIRAIS** (the stats package) | `dev/sphinx/project/libexec/config/tools/py-package/moirais/`, `r-package/moirais/` | **GPL-2.0-or-later** | Linus-style copyleft.  Derivatives must stay open.  Protects the community contribution and respects the open-source heritage we inherit from. |
| **TIDE** (terminal IDE) | `dev/sphinx/project/dev/sphinx/TIDE/` | **Apache-2.0** | User-facing tool; broader adoption helps the brand.  Apache adds patent grant. |
| **KRONOS** (terminal emulator) | `dev/sphinx/project/dev/sphinx/KRONOS/` | **Apache-2.0** | Same. |
| **wlp** (static site, landing pages, dashboards) | sibling repo `hadesllm-migration/wlp/` | **Apache-2.0** for code, **CC-BY-4.0** for prose | Public-facing; reuse with attribution is fine. |
| **keyserver** (private API) | `hadesllm-migration/wlp/keyserver/` | **All Rights Reserved** (no public license) | Billable infrastructure.  Not for redistribution.  May go BUSL later. |
| **trilogy bundle / API orchestration** | `hadesllm-migration/wlp/keyserver/`, future `/api/` endpoints | **All Rights Reserved** | Same — anything that's directly billable. |
| **pi-scripts** (operational scripts) | `hadesllm-migration/wlp/pi-scripts/` | **GPL-2.0-or-later** | Bash + Python operational glue, fits the moirais sibling license. |
| **docs prose** (Sphinx site content) | `dev/sphinx/project/docs/source/*.rst`, `*.md` | **CC-BY-4.0** | Documentation should be reusable. |

## The strategy in one paragraph

We follow the **Anthropic pattern**: the SDK / client library / public
tooling is **permissive** (MIT, Apache, or CC-BY) so it gets used
everywhere; the core stats package keeps **GPL-2.0-or-later** in
respect of the Linus / Linux heritage and to protect community
contributions; the **billing-adjacent infrastructure** (keyserver,
API orchestration) is **proprietary** ("All Rights Reserved") because
that's where the commercial model lives and it has no business being
forkable.

## Per-component LICENSE files

Each component carries its own LICENSE file at its root:

- `dev/clew/LICENSE-MIT` — MIT text
- `dev/clew/LICENSE-APACHE` — Apache 2.0 text
- `LICENSE` (repo root, default) — GPL-2.0-or-later (covers MOIRAIS + everything not otherwise licensed)
- (planned) `dev/sphinx/TIDE/LICENSE` — Apache 2.0
- (planned) `dev/sphinx/KRONOS/LICENSE` — Apache 2.0
- (planned) `hadesllm-migration/wlp/LICENSE` — Apache 2.0 (code) + `LICENSE-CONTENT` CC-BY-4.0 (prose)
- (planned) `hadesllm-migration/wlp/keyserver/LICENSE` — proprietary notice

## Contribution policy

Contributions to a permissively-licensed component (clew, TIDE,
KRONOS, wlp code, docs) are accepted under the same license as the
component (MIT/Apache/CC-BY).  Contributions to GPL components stay
GPL.  Contributions to proprietary components require a CLA (not yet
drafted; ask the author).

## Why not GPL-3?

We use **GPL-2.0**, not GPL-3.  Three reasons:

1. **Linux is GPL-2** (specifically not "or later") — it's the canonical
   choice for systems software in this lineage.
2. **GPL-3's anti-Tivoization clause** is principled but adds friction
   for embedded or signed-firmware deployments we may want later.
3. **GPL-2** is the most universally-understood copyleft license.
   Less to explain.

## Why not pure GPL for everything?

If everything were GPL, Anthropic / Google / any corp couldn't embed
clew into their tooling without GPL-ing their tooling — which they
won't do for proprietary products.  Splitting permissive (clew)
from copyleft (moirais) means we get adoption *and* community
protection where each matters.

## Why proprietary at all?

The keyserver issues API keys, manages billing relationships with
upstream providers (NASA FIRMS, WAQI, Anthropic, Google), and is the
revenue surface.  We retain rights so we can charge for it without
contractual surprises.  Source-available (BUSL) is on the table if
that becomes a stronger fit; for now, full proprietary keeps options
open.

## Acknowledgments

Compute generously provided by:

- **Anthropic** — Claude Code monthly credits used to write much of
  this code (this repo is one of the products, in a real sense).
- **Google** — Vertex AI / GenAI App Builder credits powering the
  Gemini integration in `moirais.llm`.

Influence + intellectual debt:

- **Andrej Karpathy** — for `autoresearch` (the autonomous-LLM
  training-loop fork at `dev/autoresearch-macos/`), nanoGPT,
  llm.c, and a decade of clear-thinking writing about how this
  field actually works.  The MOIRAIS engine.py + agent_loop.py
  + the entire methodology of "AI-driven research that the human
  reads, accepts, or rejects" trace directly back to his work.
  We are deeply grateful.

- **Linus Torvalds** — for git, Linux, and the GPL.  The reason
  MOIRAIS keeps GPL-2.0-or-later is in respect of this lineage.

- **Greg Wilson + the Carpentries** — for showing that
  reproducible scientific computing can be taught, not just
  demonstrated.  The "From Zero" tutorial track is in their
  spirit.

Dependencies inherit their own licenses; see each component's
manifest (`Cargo.toml`, `pyproject.toml`, `package.json`) for the
chain.

— Vansh Singh Ruhela, 2026
