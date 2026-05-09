"""Tests for moirais.fn.srinv -- serial interval estimation."""

import numpy as np
import pytest
from moirais.fn.srinv import serial_interval


class TestSerialInterval:
    def test_normal_fit(self):
        rng = np.random.default_rng(42)
        si = rng.normal(5.0, 1.5, 200)
        res = serial_interval(si, distribution="normal")
        assert res["mean"] == pytest.approx(5.0, abs=0.5)
        assert res["n"] == 200

    def test_gamma_with_negatives(self):
        rng = np.random.default_rng(42)
        si = rng.normal(4.0, 2.0, 100)
        res = serial_interval(si, distribution="gamma")
        assert res["mean"] == pytest.approx(4.0, abs=1.0)
        assert "shift" in res["params"]

    def test_prop_negative(self):
        si = np.array([-1.0, 2.0, 3.0, 5.0, -0.5])
        res = serial_interval(si, distribution="normal")
        assert res["prop_negative"] == pytest.approx(0.4)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            serial_interval(np.array([]))

    def test_ci_contains_mean(self):
        rng = np.random.default_rng(99)
        si = rng.normal(5.0, 1.0, 500)
        res = serial_interval(si, distribution="normal")
        assert res["ci_lower"] < res["mean"] < res["ci_upper"]
