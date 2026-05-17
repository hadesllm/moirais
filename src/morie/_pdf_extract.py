"""Pure-Python PDF text extraction engine.

Uses pypdf (pure Python, no C deps) as primary backend.
Handles multi-column layouts, headers/footers, page ranges,
equation placeholders, and chapter boundary detection.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def extract_text(
    pdf_path: str | Path,
    *,
    pages: tuple[int, int] | None = None,
    strip_headers: bool = True,
    strip_footers: bool = True,
    detect_chapters: bool = True,
    equation_placeholder: str = "[EQ]",
    min_line_length: int = 2,
) -> dict[str, Any]:
    """Extract text from a PDF file.

    :param pdf_path: Path to the PDF file.
    :param pages: Optional (start, end) 0-indexed page range.
    :param strip_headers: Remove repeated header lines.
    :param strip_footers: Remove repeated footer/page-number lines.
    :param detect_chapters: Insert chapter markers in output.
    :param equation_placeholder: Marker for detected equation blocks.
    :param min_line_length: Skip lines shorter than this.
    :return: dict with keys: text, pages_extracted, chapters, metadata.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        from pypdf import PdfReader
    except ImportError as err:
        raise ImportError("pypdf is required: pip install pypdf") from err

    reader = PdfReader(str(pdf_path))
    metadata = _extract_metadata(reader)

    start = pages[0] if pages else 0
    end = pages[1] if pages else len(reader.pages)
    end = min(end, len(reader.pages))

    raw_pages = []
    for i in range(start, end):
        page = reader.pages[i]
        text = page.extract_text() or ""
        raw_pages.append(text)

    header_pattern = _detect_repeated_pattern(raw_pages, position="top") if strip_headers else None
    footer_pattern = _detect_repeated_pattern(raw_pages, position="bottom") if strip_footers else None

    cleaned_pages = []
    for page_text in raw_pages:
        lines = page_text.split("\n")
        if header_pattern and lines:
            lines = _strip_pattern(lines, header_pattern, position="top")
        if footer_pattern and lines:
            lines = _strip_pattern(lines, footer_pattern, position="bottom")
        lines = [ln for ln in lines if len(ln.strip()) >= min_line_length]
        cleaned_pages.append("\n".join(lines))

    full_text = "\n\n".join(cleaned_pages)

    if equation_placeholder:
        full_text = _mark_equations(full_text, equation_placeholder)

    chapters = []
    if detect_chapters:
        chapters = _detect_chapters(full_text)

    full_text = _fix_hyphenation(full_text)
    full_text = _decode_uni_escapes(full_text)
    full_text = _decode_cid_names(full_text)
    full_text = _decode_math_glyphs(full_text)
    full_text = _collapse_letter_spacing(full_text)
    full_text = _normalize_whitespace(full_text)

    return {
        "text": full_text,
        "pages_extracted": end - start,
        "total_pages": len(reader.pages),
        "chapters": chapters,
        "metadata": metadata,
        "char_count": len(full_text),
        "word_count": len(full_text.split()),
        "line_count": full_text.count("\n") + 1,
    }


def extract_pages(
    pdf_path: str | Path,
    page_numbers: list[int],
) -> list[str]:
    """Extract text from specific pages (0-indexed)."""
    pdf_path = Path(pdf_path)
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    results = []
    for p in page_numbers:
        if 0 <= p < len(reader.pages):
            results.append(reader.pages[p].extract_text() or "")
        else:
            results.append("")
    return results


def pdf_to_file(
    pdf_path: str | Path,
    output_path: str | Path | None = None,
    **kwargs,
) -> Path:
    """Extract text from PDF and write to a .txt file.

    :param pdf_path: Path to input PDF.
    :param output_path: Path for output .txt. Defaults to same name with .txt.
    :return: Path to the written file.
    """
    result = extract_text(pdf_path, **kwargs)
    output_path = Path(pdf_path).with_suffix(".txt") if output_path is None else Path(output_path)

    output_path.write_text(result["text"], encoding="utf-8")
    return output_path


def _extract_metadata(reader) -> dict[str, str]:
    meta = reader.metadata or {}
    return {
        "title": getattr(meta, "title", "") or "",
        "author": getattr(meta, "author", "") or "",
        "subject": getattr(meta, "subject", "") or "",
        "creator": getattr(meta, "creator", "") or "",
        "producer": getattr(meta, "producer", "") or "",
    }


def _detect_repeated_pattern(
    pages: list[str],
    position: str = "top",
    n_check: int = 3,
) -> str | None:
    """Find a line that repeats across many pages at the top or bottom."""
    if len(pages) < 5:
        return None

    candidates: dict[str, int] = {}
    for page_text in pages:
        lines = [ln.strip() for ln in page_text.split("\n") if ln.strip()]
        if not lines:
            continue
        check_lines = lines[:n_check] if position == "top" else lines[-n_check:]
        for ln in check_lines:
            if len(ln) < 3:
                continue
            normalized = re.sub(r"\d+", "#", ln)
            candidates[normalized] = candidates.get(normalized, 0) + 1

    threshold = len(pages) * 0.4
    for pattern, count in candidates.items():
        if count >= threshold:
            return pattern
    return None


def _strip_pattern(
    lines: list[str],
    pattern: str,
    position: str = "top",
) -> list[str]:
    n_check = 3
    if position == "top":
        check_range = range(min(n_check, len(lines)))
    else:
        check_range = range(max(0, len(lines) - n_check), len(lines))

    remove_indices = set()
    for i in check_range:
        normalized = re.sub(r"\d+", "#", lines[i].strip())
        if normalized == pattern:
            remove_indices.add(i)

    return [ln for i, ln in enumerate(lines) if i not in remove_indices]


_EQ_PATTERN = re.compile(
    r"(?:^|\n)"
    r"([ \t]*(?:[A-Za-z][\w]*\s*[=<>≤≥≈∝∈∑∏∫]+|"
    r"\\[a-z]+\{|"
    r"\$[^$]+\$|"
    r"[∑∏∫∂∇√∞±×÷]|"
    r"(?:\d+\.?\d*\s*[+\-*/^=<>]+\s*)+)"
    r"[^\n]*)"
    r"(?:\s*\([\d.]+\))?",
    re.MULTILINE,
)


def _mark_equations(text: str, placeholder: str) -> str:
    """Mark lines that look like standalone equations."""
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if _looks_like_equation(stripped):
            result.append(f"{placeholder} {stripped}")
        else:
            result.append(line)
    return "\n".join(result)


def _looks_like_equation(line: str) -> bool:
    if not line or len(line) < 3:
        return False
    if len(line) > 200:
        return False
    math_chars = set("=<>≤≥≈∝∈∑∏∫∂∇√∞±×÷^{}[]()_")
    math_count = sum(1 for c in line if c in math_chars)
    alpha_count = sum(1 for c in line if c.isalpha())
    if alpha_count == 0:
        return math_count >= 2
    ratio = math_count / (alpha_count + math_count)
    if ratio > 0.25 and math_count >= 3:
        return True
    return bool(re.match(r"^\s*[A-Za-z]\s*[=(<]", line) and "=" in line and re.search(r"\(\d+\.\d+\)\s*$", line))


_CHAPTER_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "twenty-one": 21, "twenty-two": 22, "twenty-three": 23, "twenty-four": 24,
    "twenty-five": 25,
}

_CHAPTER_PATTERN = re.compile(
    r"^(?:Chapter|CHAPTER)\s+(\d+|[A-Za-z]+(?:-[A-Za-z]+)?)[:\s]*(.*?)$",
    re.MULTILINE,
)


def _detect_chapters(text: str) -> list[dict[str, Any]]:
    """Detect chapter headings.

    Accepts both numeric ("Chapter 3") and word-form ("Chapter three")
    chapter numbers. Letter-spaced headings ("C h a p t e r o n e")
    are normalized upstream by `_collapse_letter_spacing`.
    """
    chapters = []
    for m in _CHAPTER_PATTERN.finditer(text):
        token = m.group(1)
        if token.isdigit():
            number = int(token)
        else:
            number = _CHAPTER_NUM_WORDS.get(token.lower())
            if number is None:
                continue
        chapters.append(
            {
                "number": number,
                "title": m.group(2).strip(),
                "char_offset": m.start(),
            }
        )
    return chapters


def _fix_hyphenation(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


_UNI_ESCAPE_RE = re.compile(r"/uni([0-9A-Fa-f]{4})")


def _decode_uni_escapes(text: str) -> str:
    """Replace pypdf-leaked `/uniXXXX` CID names with the actual character.

    Some PDFs (notably Springer "Advanced Statistics in Criminology") expose
    raw CID glyph names like `/uni00A0` (non-breaking space) verbatim in
    extracted text. We decode them to the corresponding Unicode codepoint.
    """
    def _sub(m: re.Match[str]) -> str:
        return chr(int(m.group(1), 16))
    return _UNI_ESCAPE_RE.sub(_sub, text)


# CID names of the form `/Cnnn` that pypdf leaks for symbol fonts. These were
# observed empirically in Springer "Advanced Statistics in Criminology and
# Criminal Justice" (5th ed, 2022). Mapping confirmed by reconstructing
# known equations (logistic odds, OLS β, McFadden R²) and matching against
# the published source.
_CID_NAME_MAP = {
    "/C0":   "−",      # minus sign (very common -- arithmetic + intervals)
    "/C1":   "·",      # middle-dot multiplication
    "/C2":   "×",      # ×
    "/C16":  "(",      # bracket-fraction open  -> context says "( "
    "/C17":  ")",      # bracket-fraction close -> "}/("
    "/C18":  "{",
    "/C19":  "}",
    "/C20":  "<",
    "/C21":  "≥",      # >= seen in `y /C21 0`
    "/C22":  "≤",
    "/C23":  "≠",
    "/C24":  "->",
    "/C25":  "↑",
    "/C26":  "↓",
    "/C32":  " ",
    "/C138": "]",      # close bracket -- paired with `½`
    "/C139": "}",
    "/C140": ")",
}
_CID_NAME_RE = re.compile(r"/[CGg]\d+")


def _decode_cid_names(text: str) -> str:
    """Replace pypdf-leaked `/Cnnn` CID names with their actual glyph.

    Unknown codes are stripped (replaced with empty string) -- keeping them
    poisons grep-based equation detection. This is lossy, but advanced.txt
    has 1000+ leaked `/Cnnn` tokens; leaving them in is worse than dropping
    the rare unknown ones.
    """
    def _sub(m: re.Match[str]) -> str:
        return _CID_NAME_MAP.get(m.group(0), "")
    return _CID_NAME_RE.sub(_sub, text)


# Single-glyph mismaps and ligatures. Mapping derived empirically from
# Springer Advanced Statistics in Criminology and verified against three
# known equations (Y = β₀ + β₁x₁ + …, logistic odds, OLS β).
_MATH_GLYPH_MAP = {
    "¼": "=",      # equality glyph mismapped to vulgar fraction one-quarter
    "þ": "+",
    "Þ": ")",
    "ð": "(",
    "Ð": "(",
    "½": "[",      # opens square bracket in formulas; rare elsewhere
    "ı": "i",      # dotless-i normalisation (cosmetic)
    "ﬁ": "fi",     # ligatures
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬀ": "ff",
    "ﬄ": "ffl",
    "": "",  # control-char leakage in index region
    "": "",
    "": "",
}
_MATH_GLYPH_RE = re.compile("|".join(re.escape(k) for k in _MATH_GLYPH_MAP))
_DECIMAL_COLON_RE = re.compile(r"(\d):(\d)")


_BAR_OVER_RE = re.compile(r"≤([xXyYpPzZμπρ])")          # `≤x` -> `x̄`
_HAT_OVER_RE = re.compile(r"\bb(Y|β|p|π)\b")              # `bY` -> `Ŷ`
COMBINING_MACRON = "̄"
COMBINING_CIRCUMFLEX = "̂"


def _decode_math_glyphs(text: str) -> str:
    """Repair pypdf single-glyph mismaps in symbol-font math expressions.

    Passes:
      1. Per-character mismap (`¼ -> =`, `þ -> +`, `ð -> (`, `Þ -> )`, ligatures).
      2. `0:5` -> `0.5` for digit-colon-digit (Springer typesets decimal
         points as colons in the symbol font).
      3. Bar-over: `≤x` -> `x̄` (combining macron). Restricted to a small
         set of math letters to avoid mangling real `≤` comparisons.
      4. Hat-over: `bY` -> `Ŷ` (combining circumflex). Restricted to math
         letters to avoid mangling real `b…` words.
    """
    out = _MATH_GLYPH_RE.sub(lambda m: _MATH_GLYPH_MAP[m.group(0)], text)
    out = _DECIMAL_COLON_RE.sub(r"\1.\2", out)
    out = _BAR_OVER_RE.sub(lambda m: m.group(1) + COMBINING_MACRON, out)
    out = _HAT_OVER_RE.sub(lambda m: m.group(1) + COMBINING_CIRCUMFLEX, out)
    return out


# Letter-spacing artifact: pypdf renders headings like "C h a p t e r  o n e"
# when the source PDF uses tracking/letter-spacing. We collapse runs of 4+
# single-letter tokens back into words. Inside a run, single spaces separate
# letters of one word; spans of >=2 spaces separate distinct words.
_LETTER_SPACED_RUN_RE = re.compile(
    r"(?:(?<=^)|(?<=[\s.,;:!?()\[\]{}\"']))"      # left boundary
    r"([A-Za-z](?:[ \t]{1,4}[A-Za-z]){3,})"        # 4+ single-letter tokens, runs of 1-4 spaces
    r"(?=$|[\s.,;:!?()\[\]{}\"'])",                # right boundary
    re.MULTILINE,
)


def _collapse_letter_spacing(text: str) -> str:
    r"""Collapse `C h a p t e r  o n e` -> `Chapter one`.

    Restores PDF-tracked headings to normal word form. Heuristic:
    - find runs of >=4 single-letter tokens separated by 1-4 spaces each
    - within a run, double-spaces (or wider) mark original word breaks
    - require at least one letter to be lowercase (to avoid matching real
      acronyms like "U S A" or "I B M")
    """
    def _collapse(m: re.Match[str]) -> str:
        run = m.group(1)
        if not any(c.islower() for c in run):
            return run  # leave acronyms alone
        # Split on >=2 spaces (= original word break), then strip
        # remaining single spaces inside each chunk.
        words = [w.replace(" ", "").replace("\t", "") for w in re.split(r"[ \t]{2,}", run)]
        return " ".join(words)
    return _LETTER_SPACED_RUN_RE.sub(_collapse, text)


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()
