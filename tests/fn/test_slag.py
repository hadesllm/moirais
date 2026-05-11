"""Tests for morie.fn.slag -- Newton's law of cooling."""

import numpy as np
from morie.fn.slag import newton_cooling, slag
from morie.fn._containers import DescriptiveResult


class TestSlag:
    def test_alias(self):
        assert slag is newton_cooling

    def test_approaches_ambient(self):
        r = newton_cooling(100.0, 20.0, k=0.1, t_max=200)
        assert isinstance(r, DescriptiveResult)
        assert abs(r.value[-1] - 20.0) < 1.0

    def test_initial_temp(self):
        r = newton_cooling(90.0, 25.0, k=0.05)
        assert abs(r.value[0] - 90.0) < 0.1
