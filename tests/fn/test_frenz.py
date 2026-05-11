"""Tests for morie.fn.frenz -- Lyapunov exponent."""

import numpy as np
from morie.fn.frenz import lyapunov_exponent, frenz
from morie.fn._containers import DescriptiveResult


class TestFrenz:
    def test_alias(self):
        assert frenz is lyapunov_exponent

    def test_logistic_map(self):
        x = np.zeros(2000)
        x[0] = 0.1
        for i in range(1, len(x)):
            x[i] = 3.9 * x[i-1] * (1 - x[i-1])
        r = lyapunov_exponent(x, embed_dim=3, lag=1)
        assert isinstance(r, DescriptiveResult)
        assert np.isfinite(r.value)

    def test_periodic(self):
        t = np.arange(1000)
        x = np.sin(2 * np.pi * t / 50)
        r = lyapunov_exponent(x, embed_dim=2)
        assert isinstance(r, DescriptiveResult)
