"""Tests for moirais.fn.dchnt -- Ghost signal detection."""

import numpy as np
from moirais.fn.dchnt import ghost_signal, dchnt
from moirais.fn._containers import TestResult


class TestDchnt:
    def test_alias(self):
        assert dchnt is ghost_signal

    def test_has_peak(self):
        t = np.arange(200) / 100.0
        sig = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.default_rng(42).normal(0, 1, 200)
        result = ghost_signal(sig, seed=42, n_surrogates=199)
        assert isinstance(result, TestResult)
        assert result.extra["peak_frequency"] > 0

    def test_pure_noise(self):
        rng = np.random.default_rng(42)
        sig = rng.normal(0, 1, 200)
        result = ghost_signal(sig, seed=42, n_surrogates=199)
        assert result.p_value > 0.01
