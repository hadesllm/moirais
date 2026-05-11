"""Tests for morie.fn.drstr -- multiverse bootstrap."""

import numpy as np
from morie.fn.drstr import multiverse_bootstrap, drstr
from morie.fn._containers import DescriptiveResult


class TestDrstr:
    def test_alias(self):
        assert drstr is multiverse_bootstrap

    def test_mean_ci(self):
        rng = np.random.default_rng(42)
        x = rng.normal(5, 1, 100)
        r = multiverse_bootstrap(x, n_universes=500, seed=42)
        assert isinstance(r, DescriptiveResult)
        assert r.value["ci_lower"] < 5 < r.value["ci_upper"]

    def test_distribution_shape(self):
        x = np.arange(1.0, 51.0)
        r = multiverse_bootstrap(x, n_universes=200, seed=0)
        assert len(r.value["distribution"]) == 200
