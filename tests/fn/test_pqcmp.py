"""Tests for morie.fn.pqcmp — PolarQuant full compression."""

import numpy as np
import pytest

from morie.fn.pqcmp import polar_compress


class TestPolarCompress:

    def test_returns_result(self):
        x = np.random.default_rng(42).standard_normal(64)
        res = polar_compress(x, bits_mag=8, bits_dir=4)
        assert res.name == "polar_compress"
        assert res.value >= 0

    def test_compression_ratio(self):
        x = np.random.default_rng(0).standard_normal(64)
        res = polar_compress(x, bits_mag=8, bits_dir=4)
        assert res.extra["compression_ratio"] > 1.0

    def test_more_bits_lower_mse(self):
        x = np.random.default_rng(1).standard_normal(64)
        mse_low = polar_compress(x, bits_dir=2).value
        mse_high = polar_compress(x, bits_dir=8).value
        assert mse_high <= mse_low
