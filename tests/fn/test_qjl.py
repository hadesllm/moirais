"""Tests for morie.fn.qjl — QJL random projection."""

import numpy as np
import pytest

from morie.fn.qjl import qjl_project


class TestQjlProject:

    def test_returns_result(self):
        x = np.random.default_rng(42).standard_normal(128)
        res = qjl_project(x, d_target=32)
        assert res.name == "qjl_project"
        assert res.value == 32.0

    def test_sign_quantization(self):
        x = np.random.default_rng(0).standard_normal(64)
        res = qjl_project(x, d_target=16, bits=1)
        q = res.extra["quantized"]
        assert set(np.unique(q)).issubset({-1, 1})

    def test_dimensionality_reduction(self):
        x = np.random.default_rng(1).standard_normal(256)
        res = qjl_project(x, d_target=32, bits=4)
        assert res.extra["d_in"] == 256
        assert res.extra["d_target"] == 32

    def test_compression_ratio(self):
        x = np.random.default_rng(2).standard_normal(128)
        res = qjl_project(x, d_target=16, bits=1)
        assert res.extra["compression_ratio"] == pytest.approx(128 * 32 / (16 * 1))
