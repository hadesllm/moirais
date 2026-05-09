"""Tests for moirais.fn.ircnh -- MCD robust covariance."""

import numpy as np
from moirais.fn.ircnh import robust_covariance_mcd, ircnh
from moirais.fn._containers import DescriptiveResult


class TestIrcnh:
    def test_alias(self):
        assert ircnh is robust_covariance_mcd

    def test_basic_2d(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 2))
        r = robust_covariance_mcd(X, n_iter=10)
        assert isinstance(r, DescriptiveResult)
        cov = r.value
        assert cov.shape == (2, 2)
        assert cov[0, 0] > 0

    def test_with_outliers(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 2))
        X[:5] = 50.0
        r = robust_covariance_mcd(X, n_iter=20)
        assert r.value[0, 0] < 10
