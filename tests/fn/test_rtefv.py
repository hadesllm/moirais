"""Tests for morie.fn.rtefv -- Effective Rt."""

import numpy as np
from morie.fn.rtefv import rt_effective


class TestRtEffective:
    def test_basic(self):
        inc = [0, 1, 2, 4, 8, 16, 32, 20, 10, 5, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0]
        res = rt_effective(inc, window=5)
        assert "Rt_mean" in res
        assert len(res["t"]) > 0

    def test_constant_incidence(self):
        inc = [10] * 30
        res = rt_effective(inc, window=7)
        for rt in res["Rt_mean"]:
            assert rt > 0
