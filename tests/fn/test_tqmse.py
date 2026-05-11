"""Tests for morie.fn.tqmse — TurboQuant MSE-optimal quantization."""

import numpy as np
import pytest

from morie.fn.tqmse import turboquant_mse


class TestTurboquantMse:

    def test_returns_descriptive_result(self):
        x = np.random.default_rng(42).standard_normal(64)
        res = turboquant_mse(x, bits=3)
        assert res.name == "turboquant_mse"
        assert res.value >= 0

    def test_compression_ratio_3bit(self):
        x = np.random.default_rng(0).standard_normal(128)
        res = turboquant_mse(x, bits=3)
        assert res.extra["compression_ratio"] == pytest.approx(32.0 / 3)

    def test_mse_decreases_with_more_bits(self):
        x = np.random.default_rng(1).standard_normal(64)
        mse2 = turboquant_mse(x, bits=2).value
        mse4 = turboquant_mse(x, bits=4).value
        assert mse4 <= mse2

    def test_codes_shape(self):
        x = np.random.default_rng(2).standard_normal(32)
        res = turboquant_mse(x, bits=3)
        assert len(res.extra["codes"]) == 32

    def test_constant_input(self):
        x = np.ones(16)
        res = turboquant_mse(x, bits=2)
        assert res.value == pytest.approx(0.0, abs=1e-10)
