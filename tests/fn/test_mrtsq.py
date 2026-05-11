"""Tests for morie.fn.mrtsq -- Mauchly's sphericity test."""

import numpy as np
from morie.fn.mrtsq import mauchly_test, mrtsq
from morie.fn._containers import DescriptiveResult


class TestMauchlyTest:
    def test_alias(self):
        assert mrtsq is mauchly_test

    def test_returns_result(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 4))
        res = mauchly_test(X)
        assert isinstance(res, DescriptiveResult)

    def test_w_bounded(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 3))
        res = mauchly_test(X)
        assert 0 <= res.value <= 1.0 + 1e-6

    def test_has_epsilon_gg(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 4))
        res = mauchly_test(X)
        assert "epsilon_gg" in res.extra
        assert res.extra["epsilon_gg"] > 0

    def test_p_value_valid(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 3))
        res = mauchly_test(X)
        assert 0 <= res.extra["p_value"] <= 1
