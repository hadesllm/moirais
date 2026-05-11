"""Tests for morie.fn.smith -- bootstrap resampling."""

import numpy as np
from morie.fn.smith import bootstrap_resample, smith
from morie.fn._containers import DescriptiveResult


class TestSmith:
    def test_alias(self):
        assert smith is bootstrap_resample

    def test_mean_bootstrap(self):
        rng = np.random.default_rng(42)
        x = rng.normal(10, 2, 100)
        result = bootstrap_resample(x, n_boot=500, seed=42)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 10.0) < 1.0
        assert result.extra["ci_lower"] < 10.0 < result.extra["ci_upper"]

    def test_median(self):
        x = np.arange(1, 21, dtype=float)
        result = bootstrap_resample(x, stat_fn="median", n_boot=200, seed=42)
        assert abs(result.value - 10.5) < 3.0
