"""Test pdf_to_text."""
import tempfile
from pathlib import Path

import pytest

pypdf = pytest.importorskip("pypdf")

from morie.fn.pdftx import pdf_to_text, pdftx
from morie.fn._containers import DescriptiveResult


class TestPdfToText:
    def _make_test_pdf(self, tmp_path: Path) -> Path:
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        page = writer.pages[0]
        from pypdf.generic import (
            ArrayObject, ContentStream, DecodedStreamObject,
            NameObject, NumberObject,
        )
        content = b"BT /F1 12 Tf 100 700 Td (Hello World) Tj ET"
        stream = DecodedStreamObject()
        stream.set_data(content)
        page[NameObject("/Contents")] = stream
        if "/Font" not in page.get("/Resources", {}):
            from pypdf.generic import DictionaryObject
            resources = DictionaryObject()
            font_dict = DictionaryObject()
            font_obj = DictionaryObject()
            font_obj[NameObject("/Type")] = NameObject("/Font")
            font_obj[NameObject("/Subtype")] = NameObject("/Type1")
            font_obj[NameObject("/BaseFont")] = NameObject("/Helvetica")
            font_dict[NameObject("/F1")] = font_obj
            resources[NameObject("/Font")] = font_dict
            page[NameObject("/Resources")] = resources
        pdf_path = tmp_path / "test.pdf"
        with open(pdf_path, "wb") as f:
            writer.write(f)
        return pdf_path

    def test_basic(self, tmp_path):
        pdf_path = self._make_test_pdf(tmp_path)
        result = pdf_to_text(str(pdf_path))
        assert isinstance(result, DescriptiveResult)
        assert result.name == "pdf_to_text"
        assert result.extra["pages_extracted"] >= 1
        assert result.extra["total_pages"] >= 1
        assert isinstance(result.extra["text"], str)

    def test_output_type(self, tmp_path):
        pdf_path = self._make_test_pdf(tmp_path)
        result = pdf_to_text(str(pdf_path))
        assert "word_count" in result.extra
        assert "char_count" in result.extra
        assert "metadata" in result.extra

    def test_alias(self):
        assert pdftx is pdf_to_text
