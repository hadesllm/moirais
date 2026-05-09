"""Tests for moirais.fn.rtn — round-to-nearest quantizer."""

import numpy as np
import pytest

from moirais.fn.rtn import round_to_nearest


class TestRoundToNearest:

    def test_returns_result(self):
        x = np.random.default_rng(42).standard_normal(64)
        res = round_to_nearest(x, bits=4)
        assert res.name == "round_to_nearest"
        assert res.value >= 0

    def test_higher_bits_lower_mse(self):
        x = np.random.default_rng(0).standard_normal(100)
        mse2 = round_to_nearest(x, bits=2).value
        mse8 = round_to_nearest(x, bits=8).value
        assert mse8 < mse2

    def test_constant_input(self):
        x = np.ones(16) * 5.0
        res = round_to_nearest(x, bits=4)
        assert res.value == pytest.approx(0.0, abs=1e-10)
