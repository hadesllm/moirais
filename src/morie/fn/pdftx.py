# morie.fn -- function file (hadesllm/morie)
"""PDF text extraction."""

from __future__ import annotations

from ._containers import DescriptiveResult


def pdf_to_text(
    pdf_path: str,
    *,
    pages: tuple[int, int] | None = None,
    output_path: str | None = None,
    strip_headers: bool = True,
    strip_footers: bool = True,
    detect_chapters: bool = True,
) -> DescriptiveResult:
    from morie._pdf_extract import extract_text, pdf_to_file

    result = extract_text(
        pdf_path,
        pages=pages,
        strip_headers=strip_headers,
        strip_footers=strip_footers,
        detect_chapters=detect_chapters,
    )

    out_path = None
    if output_path is not None:
        out_path = str(
            pdf_to_file(
                pdf_path,
                output_path=output_path,
                strip_headers=strip_headers,
                strip_footers=strip_footers,
                detect_chapters=detect_chapters,
            )
        )

    return DescriptiveResult(
        name="pdf_to_text",
        value=result["word_count"],
        extra={
            "text": result["text"],
            "pages_extracted": result["pages_extracted"],
            "total_pages": result["total_pages"],
            "chapters": result["chapters"],
            "metadata": result["metadata"],
            "char_count": result["char_count"],
            "word_count": result["word_count"],
            "line_count": result["line_count"],
            "output_path": out_path,
        },
    )


pdftx = pdf_to_text


def cheatsheet() -> str:
    return "pdf_to_text({}) -> PDF text extraction."
