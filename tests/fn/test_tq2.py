"""Tests for moirais.fn.tq2 — 2-bit TurboQuant."""

import numpy as np
import pytest

from moirais.fn.tq2 import turboquant_2bit


class TestTurboquant2bit:

    def test_returns_result(self):
        x = np.random.default_rng(42).standard_normal(64)
        res = turboquant_2bit(x)
        assert res.name == "turboquant_mse"
        assert res.extra["bits"] == 2

    def test_target_compression(self):
        x = np.random.default_rng(0).standard_normal(32)
        res = turboquant_2bit(x)
        assert res.extra["target_compression"] == 14.6

    def test_compression_ratio(self):
        x = np.random.default_rng(1).standard_normal(64)
        res = turboquant_2bit(x)
        assert res.extra["compression_ratio"] == pytest.approx(16.0)
