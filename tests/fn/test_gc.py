"""Tests for morie.fn.gc -- GC content."""

import pytest
from morie.fn.gc import gc_content_calc


class TestGCContent:
    def test_mixed_sequence(self):
        """ATGCGC has 4 GC out of 6 bases."""
        res = gc_content_calc("ATGCGC")
        assert res.measure == "GC_content"
        assert res.estimate == pytest.approx(4 / 6, abs=1e-10)
        assert res.n == 6

    def test_all_at(self):
        """Sequence with no G or C has GC = 0."""
        res = gc_content_calc("AAAA")
        assert res.estimate == pytest.approx(0.0, abs=1e-10)

    def test_all_gc(self):
        """Sequence of only G and C has GC = 1."""
        res = gc_content_calc("GGGCCC")
        assert res.estimate == pytest.approx(1.0, abs=1e-10)

    def test_case_insensitive(self):
        """Lowercase input should work."""
        res = gc_content_calc("atgcgc")
        assert res.estimate == pytest.approx(4 / 6, abs=1e-10)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            gc_content_calc("")
