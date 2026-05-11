"""Tests for morie.fn.boxmt -- Box's M test."""

import numpy as np
from morie.fn.boxmt import box_m_test, boxmt
from morie.fn._containers import DescriptiveResult


class TestBoxMTest:
    def test_alias(self):
        assert boxmt is box_m_test

    def test_returns_result(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 1, (30, 3)), rng.normal(0, 1, (30, 3))])
        g = np.array([0]*30 + [1]*30)
        res = box_m_test(X, g)
        assert isinstance(res, DescriptiveResult)

    def test_equal_covariances_not_significant(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (60, 2))
        g = np.array([0]*30 + [1]*30)
        res = box_m_test(X, g)
        assert res.extra["p_value"] > 0.01

    def test_unequal_covariances_significant(self):
        rng = np.random.default_rng(42)
        X1 = rng.normal(0, 1, (40, 3))
        X2 = rng.normal(0, 5, (40, 3))
        X = np.vstack([X1, X2])
        g = np.array([0]*40 + [1]*40)
        res = box_m_test(X, g)
        assert res.extra["p_value"] < 0.05
