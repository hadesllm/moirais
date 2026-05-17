"""Tests for morie.fn.cepana -- cepstral analysis."""

import numpy as np
from morie.fn.cepana import cepstral_analysis, cepana
from morie.fn._containers import DescriptiveResult


class TestCepana:
    def test_alias(self):
        assert cepana is cepstral_analysis

    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 256)
        r = cepstral_analysis(x, n_coeffs=13)
        assert isinstance(r, DescriptiveResult)
        assert len(r.value) == 13

    def test_pure_tone(self):
        t = np.arange(512) / 8000.0
        x = np.sin(2 * np.pi * 440 * t)
        r = cepstral_analysis(x, fs=8000, n_coeffs=20)
        assert len(r.value) == 20
