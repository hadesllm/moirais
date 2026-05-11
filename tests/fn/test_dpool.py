"""Tests for morie.fn.dpool -- jackknife delete-d."""

import numpy as np
from morie.fn.dpool import jackknife_delete_d, dpool
from morie.fn._containers import DescriptiveResult


class TestDpool:
    def test_alias(self):
        assert dpool is jackknife_delete_d

    def test_mean(self):
        x = np.arange(1.0, 21.0)
        r = jackknife_delete_d(x, d=1)
        assert isinstance(r, DescriptiveResult)
        assert abs(r.value["estimate"] - 10.5) < 0.01

    def test_delete_d(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 20)
        r = jackknife_delete_d(x, d=3, seed=42)
        assert r.value["se"] > 0
        assert r.value["n_replicates"] > 0
