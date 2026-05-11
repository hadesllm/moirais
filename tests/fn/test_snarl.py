"""Tests for morie.fn.snarl -- resonance Q-factor."""

import numpy as np
from morie.fn.snarl import resonance_q, snarl
from morie.fn._containers import DescriptiveResult


class TestSnarl:
    def test_alias(self):
        assert snarl is resonance_q

    def test_sharp_resonance(self):
        f = np.linspace(1, 200, 1000)
        f0, Q_true = 100, 50
        a = 1.0 / np.sqrt((1 - (f / f0) ** 2) ** 2 + (f / (Q_true * f0)) ** 2)
        r = resonance_q(f, a)
        assert isinstance(r, DescriptiveResult)
        assert abs(r.extra["f_resonance"] - 100) < 5

    def test_flat_response(self):
        f = np.linspace(1, 100, 500)
        a = np.ones_like(f)
        a[250] = 1.01
        r = resonance_q(f, a)
        assert r.value > 0
