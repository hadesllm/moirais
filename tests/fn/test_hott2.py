"""Tests for morie.fn.hott2 -- Hotelling's T-squared test."""

import numpy as np
from morie.fn.hott2 import hotelling_t2, hott2
from morie.fn._containers import DescriptiveResult


class TestHotellingT2:
    def test_alias(self):
        assert hott2 is hotelling_t2

    def test_one_sample(self):
        rng = np.random.default_rng(42)
        X = rng.normal(3, 1, (40, 3))
        res = hotelling_t2(X)
        assert isinstance(res, DescriptiveResult)
        assert res.value > 0

    def test_one_sample_significant(self):
        rng = np.random.default_rng(42)
        X = rng.normal(5, 1, (50, 2))
        res = hotelling_t2(X)
        assert res.extra["p_value"] < 0.05

    def test_two_sample(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (30, 3))
        Y = rng.normal(3, 1, (30, 3))
        res = hotelling_t2(X, Y)
        assert res.extra["p_value"] < 0.05

    def test_two_sample_same_not_significant(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (30, 3))
        Y = rng.normal(0, 1, (30, 3))
        res = hotelling_t2(X, Y)
        assert res.extra["p_value"] > 0.01
