# Licensing

MOIRAIS is released under the **GNU General Public License, version 2 only**
(`GPL-2.0-only`). The full license text is in [`LICENSE`](LICENSE).

## Summary

| Component | Path | License |
|---|---|---|
| Python package | `src/moirais/` | **GPL-2.0-only** |
| R package | `r-package/moirais/` | **GPL-2.0-only** |
| Documentation prose | `docs/source/*.rst`, `*.md` | **CC-BY-4.0** |
| Paper text and figures | `paper.md`, `paper.bib` | **CC-BY-4.0** |
| Test data and fixtures | `tests/`, `src/moirais/data/` | **GPL-2.0-only** |

## Why GPL-2.0-only (not "or later")

1. **Linux precedent.** The Linux kernel is `GPL-2.0-only` — Linus chose
   v2-only deliberately, and MOIRAIS keeps that choice in the same
   spirit.
2. **No automatic upgrade.** "or later" delegates the licensing decision
   to a future Free Software Foundation. Pinning to v2 keeps the licence
   that contributors actually agreed to.
3. **GPL-3's anti-Tivoization clause** is principled but adds friction
   for embedded and signed-firmware deployments. v2 keeps options open.
4. **Universally understood.** GPL-2 is the most widely interpreted
   copyleft licence; less explanation, fewer edge cases.

## What this means in practice

- You may use, modify, and redistribute MOIRAIS under GPL-2.0-only.
- If you distribute a derivative work, it must also be released under
  GPL-2.0-only and ship complete corresponding source.
- Linking MOIRAIS into a non-GPL program is not permitted; depending on
  MOIRAIS as an external runtime (e.g. calling `python -m moirais`) is.
- The package itself imposes no restriction on the *output* of analyses —
  research results produced with MOIRAIS are yours to license as you
  choose.

## Documentation and paper text

Prose content (`docs/source/`, `paper.md`, `README.md`, this file) is
released under **Creative Commons Attribution 4.0 International**
(`CC-BY-4.0`). You may reuse it with attribution.

## Dependencies

Each dependency carries its own licence; see `pyproject.toml`
(Python) and `r-package/moirais/DESCRIPTION` (R) for the chain.

## Contributing

Contributions to MOIRAIS are accepted under `GPL-2.0-only`. By
submitting a pull request you agree to license your contribution under
that licence.

## Reporting licence issues

Open an issue at <https://github.com/hadesllm/moirais/issues> or contact
the author.

— Vansh Singh Ruhela, 2026
