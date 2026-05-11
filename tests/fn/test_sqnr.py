"""Tests for morie.fn.sqnr — signal-to-quantization-noise ratio."""

import numpy as np
import pytest

from morie.fn.sqnr import signal_quant_noise_ratio


class TestSqnr:

    def test_perfect_reconstruction(self):
        x = np.random.default_rng(42).standard_normal(64)
        res = signal_quant_noise_ratio(x, x)
        assert res.value == float("inf")

    def test_positive_for_noisy(self):
        x = np.random.default_rng(0).standard_normal(64)
        x_hat = x + np.random.default_rng(1).standard_normal(64) * 0.01
        res = signal_quant_noise_ratio(x, x_hat)
        assert res.value > 0

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            signal_quant_noise_ratio(np.ones(10), np.ones(20))

    def test_returns_db(self):
        x = np.ones(100)
        x_hat = x + 0.1
        res = signal_quant_noise_ratio(x, x_hat)
        assert res.extra["sqnr_db"] == res.value
