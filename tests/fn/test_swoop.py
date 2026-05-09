"""Tests for moirais.fn.swoop -- lift-drag polar."""

import numpy as np
from moirais.fn.swoop import lift_drag_polar, swoop
from moirais.fn._containers import DescriptiveResult


class TestSwoop:
    def test_alias(self):
        assert swoop is lift_drag_polar

    def test_basic_polar(self):
        cl = np.linspace(-0.5, 1.5, 50)
        cd = 0.02 + 0.04 * cl ** 2
        r = lift_drag_polar(cl, cd)
        assert isinstance(r, DescriptiveResult)
        assert r.value > 0
        assert r.extra["Cd0"] > 0

    def test_ld_max_positive(self):
        cl = np.linspace(0.1, 2.0, 30)
        cd = 0.01 + 0.05 * cl ** 2
        r = lift_drag_polar(cl, cd)
        assert r.value > 5
