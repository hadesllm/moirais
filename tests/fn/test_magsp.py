"""Test magnitude_spectrum (magsp)."""
import numpy as np
from moirais.fn.magsp import magnitude_spectrum, magsp
from moirais.fn._containers import DescriptiveResult


class TestMagsp:
    def test_basic(self):
        x = np.array([1.0, 0.0, -1.0, 0.0])
        result = magnitude_spectrum(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "magnitude_spectrum"

    def test_positive_values(self):
        x = np.random.default_rng(42).standard_normal(32)
        result = magnitude_spectrum(x)
        assert np.all(result.extra["magnitude"] >= 0)

    def test_alias(self):
        assert magsp is magnitude_spectrum
