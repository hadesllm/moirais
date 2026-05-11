"""Tests for morie.fn.dblrt -- population doubling time."""

import pytest
import numpy as np
from morie.fn.dblrt import population_doubling


class TestDoublingTime:
    def test_basic(self):
        res = population_doubling(growth_rate=0.02)
        assert res.estimate == pytest.approx(np.log(2) / 0.02)

    def test_zero_growth(self):
        res = population_doubling(0.0)
        assert res.estimate == float("inf")
