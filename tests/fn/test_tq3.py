"""Tests for moirais.fn.tq3 — 3-bit TurboQuant."""

import numpy as np
import pytest

from moirais.fn.tq3 import turboquant_3bit


class TestTurboquant3bit:

    def test_returns_result(self):
        x = np.random.default_rng(42).standard_normal(64)
        res = turboquant_3bit(x)
        assert res.extra["bits"] == 3

    def test_target_compression(self):
        x = np.random.default_rng(0).standard_normal(32)
        res = turboquant_3bit(x)
        assert res.extra["target_compression"] == 10.0
