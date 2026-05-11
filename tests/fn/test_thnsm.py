"""Tests for morie.fn.thnsm -- random half-sampling estimator."""

import numpy as np
from morie.fn.thnsm import snap_estimator, thnsm
from morie.fn._containers import DescriptiveResult


class TestThnsm:
    def test_alias(self):
        assert thnsm is snap_estimator

    def test_mean_estimate(self):
        rng = np.random.default_rng(42)
        x = rng.normal(10, 2, 100)
        r = snap_estimator(x, statistic="mean", n_snaps=500, seed=42)
        assert isinstance(r, DescriptiveResult)
        assert abs(r.value["estimate"] - 10) < 1

    def test_median(self):
        x = np.arange(1.0, 101.0)
        r = snap_estimator(x, statistic="median", seed=0)
        assert r.value["ci_lower"] < r.value["ci_upper"]
