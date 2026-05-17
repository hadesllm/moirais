"""Tests for morie.fn.prores -- bootstrap quantile resampling."""

import numpy as np
from morie.fn.prores import probability_resample, prores
from morie.fn._containers import DescriptiveResult


class TestProres:
    def test_alias(self):
        assert prores is probability_resample

    def test_median(self):
        rng = np.random.default_rng(42)
        x = rng.normal(10, 2, 100)
        r = probability_resample(x, target_quantile=0.5, n_resamples=500, seed=42)
        assert isinstance(r, DescriptiveResult)
        assert abs(r.value["estimate"] - 10) < 2
        assert r.value["ci_lower"] < r.value["ci_upper"]

    def test_se_positive(self):
        x = np.arange(1.0, 51.0)
        r = probability_resample(x, seed=0)
        assert r.value["se"] > 0
