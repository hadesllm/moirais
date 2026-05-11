"""Tests for morie.fn.semap -- Sammon mapping."""

import numpy as np
from morie.fn.semap import sammon_mapping, semap
from morie.fn._containers import DescriptiveResult


class TestSammonMapping:
    def test_alias(self):
        assert semap is sammon_mapping

    def test_returns_result(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 4))
        res = sammon_mapping(X, n_dims=2)
        assert isinstance(res, DescriptiveResult)

    def test_shape(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 4))
        res = sammon_mapping(X, n_dims=2)
        assert res.value.shape == (20, 2)

    def test_stress_nonneg(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((15, 3))
        res = sammon_mapping(X, n_dims=2)
        assert res.extra["stress"] >= 0
