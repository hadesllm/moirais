"""Tests for moirais.fn.nuc — nucleotide frequency."""
import pytest
from moirais.fn.nuc import nucleotide_freq


class TestNucleotideFreq:
    def test_basic_counts(self):
        res = nucleotide_freq("AATTGGCC")
        assert res.extra["counts"]["A"] == 2
        assert res.extra["counts"]["T"] == 2
        assert res.extra["gc_content"] == pytest.approx(0.5)

    def test_gc_content(self):
        res = nucleotide_freq("GGGG")
        assert res.value == pytest.approx(1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            nucleotide_freq("")
