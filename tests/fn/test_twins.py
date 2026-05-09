"""Tests for moirais.fn.twins -- phase locking value."""

import numpy as np
from moirais.fn.twins import phase_locking_value, twins
from moirais.fn._containers import DescriptiveResult


class TestTwins:
    def test_alias(self):
        assert twins is phase_locking_value

    def test_identical_signals(self):
        t = np.linspace(0, 1, 500)
        x = np.sin(2 * np.pi * 10 * t)
        result = phase_locking_value(x, x, fs=500)
        assert isinstance(result, DescriptiveResult)
        assert result.value > 0.99

    def test_independent(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        y = rng.normal(0, 1, 500)
        result = phase_locking_value(x, y)
        assert result.value < 0.3
