"""Tests for moirais.fn.arcee -- arc length computation."""

import numpy as np
from moirais.fn.arcee import arc_length, arcee
from moirais.fn._containers import DescriptiveResult


class TestArcee:
    def test_alias(self):
        assert arcee is arc_length

    def test_straight_line(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.zeros(4)
        r = arc_length(x, y)
        assert isinstance(r, DescriptiveResult)
        assert abs(r.value - 3.0) < 1e-10

    def test_circle_quarter(self):
        t = np.linspace(0, np.pi / 2, 10000)
        x = np.cos(t)
        y = np.sin(t)
        r = arc_length(x, y)
        assert abs(r.value - np.pi / 2) < 0.01
