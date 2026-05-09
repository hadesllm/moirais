"""Tests for moirais.fn.codeg -- Wigner semicircle law."""

from moirais.fn.codeg import wigner_semicircle, codeg
from moirais.fn._containers import DescriptiveResult


class TestCodeg:
    def test_alias(self):
        assert codeg is wigner_semicircle

    def test_ks_small(self):
        result = wigner_semicircle(n=200, seed=42)
        assert isinstance(result, DescriptiveResult)
        assert result.value < 0.2

    def test_larger_matrix(self):
        result = wigner_semicircle(n=500, seed=42)
        assert result.value < result.extra["theoretical_radius"]
