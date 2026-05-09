"""Tests for moirais.fn.sludg -- Herschel-Bulkley rheology."""

import numpy as np
from moirais.fn.sludg import herschel_bulkley, sludg
from moirais.fn._containers import DescriptiveResult


class TestSludg:
    def test_alias(self):
        assert sludg is herschel_bulkley

    def test_newtonian(self):
        sr = np.linspace(0.1, 100, 50)
        ss = 0 + 1.0 * sr ** 1.0 + np.random.default_rng(42).normal(0, 0.1, 50)
        r = herschel_bulkley(sr, ss)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["r_squared"] > 0.9

    def test_shear_thinning(self):
        sr = np.linspace(0.1, 100, 50)
        ss = 5 + 2.0 * sr ** 0.5
        r = herschel_bulkley(sr, ss)
        assert r.extra["behavior"] == "shear-thinning"
