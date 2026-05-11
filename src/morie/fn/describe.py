# morie.fn — function file (hadesllm/morie)
"""describe(name) — pedagogical multi-section guide for any morie.fn callable.

Per  and the rule in
memory `feedback_morie_describe_feature.md`. Reads a sibling
`describe_<short>.md` file, parses it into 9 standard sections, and
returns a `RichResult` for verbose multi-paragraph output.
"""

from __future__ import annotations

import re
from pathlib import Path

_FN_DIR = Path(__file__).parent
_DESCRIBE_PREFIX = "describe_"

# The 9 standard section headers we expect in describe_<name>.md files.
# Order matters — this is the order they're rendered.
_STANDARD_SECTIONS = [
    "WHAT IT DOES",
    "WHEN TO USE",
    "WHEN NOT TO USE",
    "ASSUMPTIONS",
    "FORMULA",
    "INPUTS / OUTPUTS",
    "WORKED EXAMPLE",
    "COMMON MISTAKES",
    "REFERENCES",
]


def describe(name: str):
    """Pedagogical guide for a morie.fn callable.

    Looks up `name` in REGISTRY, loads the sibling describe_<name>.md
    file, parses it into 9 sections, returns a `RichResult`.

    When the .md file is missing, builds a minimal skeleton from the
    REGISTRY metadata + docstring with a "(full guide pending)" header.

    >>> from morie.fn import describe
    >>> print(describe("welcht"))
    """
    from ._richresult import RichResult
    from ._registry import REGISTRY

    name = name.lstrip("/")  # tolerant of "/welcht" etc.
    entry = REGISTRY.get(name)
    if entry is None:
        # Try case-insensitive match
        match = next((REGISTRY[k] for k in REGISTRY if k.lower() == name.lower()), None)
        if match is not None:
            entry = match
        else:
            # Fallback: if the fn module exists and a describe_<name>.md is
            # next to it, synthesize a minimal entry from the docstring.
            # This keeps auto-generated callables (no REGISTRY entry yet)
            # describable.
            entry = _synthesize_entry_from_module(name)
            if entry is None:
                avail = ", ".join(sorted(REGISTRY.keys())[:20])
                raise ValueError(
                    f"unknown callable: {name!r}. Try one of: {avail}, …  "
                    f"(REGISTRY has {len(REGISTRY)} entries)."
                )

    # Find the markdown file
    md_path = _FN_DIR / f"{_DESCRIBE_PREFIX}{entry.short}.md"
    if md_path.exists():
        md_text = md_path.read_text(encoding="utf-8")
        return _render_from_md(md_text, entry, md_path)
    return _render_skeleton(entry)


def _synthesize_entry_from_module(name: str):
    """Build a minimal REGISTRY-compatible entry for a generated callable
    that doesn't yet have a hand-curated REGISTRY row.
    Returns None if the fn/<name>.py file doesn't exist.
    """
    py_path = _FN_DIR / f"{name}.py"
    if not py_path.exists():
        return None
    from ._registry import FnEntry
    text = py_path.read_text(encoding="utf-8", errors="replace")
    # First docstring line is the description
    desc = ""
    for line in text.splitlines():
        s = line.strip().strip('"""').strip()
        if s and not s.startswith(("import", "from", "__all__")):
            desc = s
            break
    return FnEntry(
        short=name,
        full=name,
        category="auto-generated",
        description=desc or f"Auto-generated callable {name}",
        quote="",
    )


def _render_from_md(md_text: str, entry, md_path: Path):
    """Parse the markdown into sections and build a RichResult."""
    from ._richresult import RichResult

    sections = _parse_sections(md_text)

    rich_sections = []
    for label in _STANDARD_SECTIONS:
        body = sections.get(label, "")
        if body.strip():
            rich_sections.append({
                "title": label,
                "text": body.strip(),
            })

    return RichResult(
        title=f"describe({entry.short!r}) — {entry.description or entry.full}",
        summary_lines=[
            ("Short name", entry.short),
            ("Category", entry.category),
            ("Full name", entry.full),
        ],
        sections=rich_sections,
        extras=[f"Source: {md_path.name}"],
        payload={"name": entry.short, "full_name": entry.full,
                 "category": entry.category,
                 "description": entry.description,
                 "sections": sections},
    )


def _render_skeleton(entry):
    """Fallback: build a minimal description from REGISTRY metadata when
    no describe_*.md file exists yet."""
    from ._richresult import RichResult

    return RichResult(
        title=f"describe({entry.short!r}) — {entry.full}",
        summary_lines=[
            ("Short name", entry.short),
            ("Category", entry.category),
            ("Description", entry.description),
            ("Quote", entry.quote or "—"),
        ],
        sections=[{
            "title": "WHAT IT DOES",
            "text": (entry.description or "(no description provided)") + "\n\n"
                    "We are what we repeatedly do. Excellence is a habit. — Aristotle"
                    f"`describe_{entry.short}.md` next to its source file.)",
        }],
        warnings=[
            f"No describe_{entry.short}.md file found. Showing skeleton "
            "from REGISTRY metadata only.",
        ],
        payload={"name": entry.short, "full_name": entry.full,
                 "category": entry.category,
                 "description": entry.description,
                 "is_skeleton": True},
    )


_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.M)


def _parse_sections(md_text: str) -> dict[str, str]:
    """Split markdown into sections keyed by ## headers.

    Returns a dict of {header_text: body}.
    """
    matches = list(_HEADING_RE.finditer(md_text))
    sections = {}
    for i, m in enumerate(matches):
        header = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        body = md_text[start:end].strip()
        sections[header] = body
    return sections
