"""Tests for morie.fn.dblng -- doubling time."""

import numpy as np
import pytest
from morie.fn.dblng import doubling_time


class TestDoublingTime:
    def test_from_growth_rate(self):
        res = doubling_time(growth_rate=0.1)
        assert res["doubling_time"] == pytest.approx(np.log(2) / 0.1, rel=1e-6)

    def test_negative_growth(self):
        res = doubling_time(growth_rate=-0.05)
        assert res["doubling_time"] == np.inf

    def test_from_incidence(self):
        t = np.arange(30)
        inc = 10 * np.exp(0.15 * t)
        res = doubling_time(incidence=inc)
        assert res["doubling_time"] == pytest.approx(np.log(2) / 0.15, abs=0.5)

    def test_no_input_raises(self):
        with pytest.raises(ValueError):
            doubling_time()

    def test_zero_growth(self):
        res = doubling_time(growth_rate=0.0)
        assert res["doubling_time"] == np.inf
