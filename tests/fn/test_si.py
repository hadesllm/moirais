"""Tests for moirais.fn.si — Serial interval estimation."""

import numpy as np
import pytest

from moirais.fn.si import serial_interval


class TestSerialInterval:
    def test_basic(self):
        rng = np.random.default_rng(42)
        primary = rng.uniform(0, 10, 50)
        secondary = primary + rng.gamma(3, 1.5, 50)
        res = serial_interval(primary, secondary)
        assert res.estimate > 0
        assert res.n == 50

    def test_ci_contains_estimate(self):
        rng = np.random.default_rng(42)
        primary = rng.uniform(0, 10, 30)
        secondary = primary + rng.gamma(3, 1.5, 30)
        res = serial_interval(primary, secondary)
        assert res.ci_lower <= res.estimate <= res.ci_upper

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            serial_interval([1, 2], [1])
