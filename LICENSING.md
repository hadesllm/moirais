# Licensing

MORIE is released under the **GNU General Public License, version 2 only**
(`GPL-2.0-only`). The full license text is in [`LICENSE`](LICENSE).

## Summary

| Component | Path | License |
|---|---|---|
| Python package | `src/morie/` | **GPL-2.0-only** |
| R package | `r-package/morie/` | **GPL-2.0-only** |
| Documentation prose | `docs/source/*.rst`, `*.md` | **CC-BY-4.0** |
| Paper text and figures | `paper.md`, `paper.bib` | **CC-BY-4.0** |
| Test data and fixtures | `tests/`, `src/morie/data/` | **GPL-2.0-only** |

## Why GPL-2.0-only (not "or later")

1. **Linux precedent.** The Linux kernel is `GPL-2.0-only` — Linus chose
   v2-only deliberately, and MORIE keeps that choice in the same
   spirit.
2. **No automatic upgrade.** "or later" delegates the licensing decision
   to a future Free Software Foundation. Pinning to v2 keeps the licence
   that contributors actually agreed to.
3. **GPL-3's anti-Tivoization clause** is principled but adds friction
   for embedded and signed-firmware deployments. v2 keeps options open.
4. **Universally understood.** GPL-2 is the most widely interpreted
   copyleft licence; less explanation, fewer edge cases.

## What this means in practice

- You may use, modify, and redistribute MORIE under GPL-2.0-only.
- If you distribute a derivative work, it must also be released under
  GPL-2.0-only and ship complete corresponding source.
- Linking MORIE into a non-GPL program is not permitted; depending on
  MORIE as an external runtime (e.g. calling `python -m morie`) is.
- The package itself imposes no restriction on the *output* of analyses —
  research results produced with MORIE are yours to license as you
  choose.

## Documentation and paper text

Prose content (`docs/source/`, `paper.md`, `README.md`, this file) is
released under **Creative Commons Attribution 4.0 International**
(`CC-BY-4.0`). You may reuse it with attribution.

## Dependencies

Each dependency carries its own licence; see `pyproject.toml`
(Python) and `r-package/morie/DESCRIPTION` (R) for the chain.

## AI assistance and third-party tools

The MORIE source was developed with substantial assistance from
frontier AI assistants. This is disclosed for transparency; it does
not transfer authorship, copyright, or licensing obligations away
from the human author. The author retains full responsibility for
the code, the methods, and the scientific claims.

- **Anthropic Claude.** Anthropic's Claude family (Opus, Sonnet, and
  Haiku across the 4.x generation) was used extensively for code
  generation, refactoring, documentation, code review, and design
  discussions. Use was supported by Anthropic research-credit
  programs.

- **Google Gemini and Vertex AI.** Google's Gemini 2.5 models (Pro and
  Flash) on the Vertex AI platform were used extensively for additional
  code generation, cross-checking Claude-generated code, multi-modal
  data analysis, and prototype evaluation. Use was supported by Google
  research-credit programs.

The Anthropic and Google research-credit programs are compute-
allocation programs; they do not constitute endorsement of MORIE
by either company. Where AI-generated code reproduces material from
training data verbatim, the upstream licence governs that material.
The author has reviewed the source for any such cases.

## Contributing

Contributions to MORIE are accepted under `GPL-2.0-only`. By
submitting a pull request you agree to license your contribution under
that licence.

## Reporting licence issues

Open an issue at <https://github.com/hadesllm/morie/issues> or contact
the author.

— Vansh Singh Ruhela, 2026
