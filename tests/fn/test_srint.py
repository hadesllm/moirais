"""Tests for morie.fn.srint -- Serial interval estimation."""

import pytest
import numpy as np
from morie.fn.srint import serial_interval


class TestSerialInterval:
    def test_gamma(self):
        rng = np.random.default_rng(42)
        data = rng.gamma(shape=3, scale=1.5, size=100)
        res = serial_interval(data, distribution="gamma")
        assert res.measure == "serial_interval"
        assert res.estimate > 0

    def test_ci_brackets(self):
        rng = np.random.default_rng(42)
        data = rng.gamma(shape=3, scale=1.5, size=200)
        res = serial_interval(data)
        assert res.ci_lower < res.estimate < res.ci_upper

    def test_too_few(self):
        with pytest.raises(ValueError):
            serial_interval([1.0, 2.0])
