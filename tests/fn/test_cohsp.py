"""Test coherence_spectrum (cohsp)."""
import numpy as np
import pytest

from morie.fn.cohsp import coherence_spectrum, cohsp
from morie.fn._containers import DescriptiveResult


class TestCoherenceSpectrum:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(512)
        y = rng.standard_normal(512)
        result = coherence_spectrum(x, y, fs=100.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "coherence_spectrum"

    def test_self_coherence(self):
        x = np.random.default_rng(42).standard_normal(512)
        result = coherence_spectrum(x, x, fs=100.0)
        assert result.value == pytest.approx(1.0, abs=0.01)

    def test_extra_keys(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(512)
        y = rng.standard_normal(512)
        result = coherence_spectrum(x, y, fs=100.0)
        assert "frequencies" in result.extra
        assert "coherence" in result.extra

    def test_alias(self):
        assert cohsp is coherence_spectrum
