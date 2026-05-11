"""Tests for morie.fn.unfrm — uniform quantizer."""

import numpy as np
import pytest

from morie.fn.unfrm import uniform_quantize


class TestUniformQuantize:

    def test_returns_result(self):
        x = np.random.default_rng(42).standard_normal(100)
        res = uniform_quantize(x, bits=8)
        assert res.name == "uniform_quantize"

    def test_8bit_low_mse(self):
        x = np.random.default_rng(0).standard_normal(100)
        res = uniform_quantize(x, bits=8)
        assert res.value < 0.01

    def test_codes_range(self):
        x = np.random.default_rng(1).standard_normal(50)
        res = uniform_quantize(x, bits=4)
        assert np.all(res.extra["codes"] >= 0)
        assert np.all(res.extra["codes"] < 16)
