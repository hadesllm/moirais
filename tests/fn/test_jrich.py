"""Tests for morie.fn.jrich -- Formant extraction."""

import numpy as np
from morie.fn.jrich import formant_extract, jrich
from morie.fn._containers import DescriptiveResult


class TestJrich:
    def test_alias(self):
        assert jrich is formant_extract

    def test_synthetic_vowel(self):
        sr = 16000
        t = np.arange(sr) / sr
        sig = np.sin(2 * np.pi * 500 * t) + 0.5 * np.sin(2 * np.pi * 1500 * t)
        result = formant_extract(sig, sr=sr, order=12)
        assert isinstance(result, DescriptiveResult)
        assert len(result.value) > 0

    def test_order_param(self):
        rng = np.random.default_rng(42)
        sig = rng.normal(0, 1, 1000)
        result = formant_extract(sig, sr=8000, order=8, n_formants=2)
        assert result.extra["order"] == 8
