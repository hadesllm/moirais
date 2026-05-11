"""Tests for morie.fn.hc012 — HC0/HC1/HC2/HC3 robust SE."""

import numpy as np
import pytest

from morie.fn.hc012 import hc_robust_se


@pytest.fixture()
def het_data():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 1))
    sigma = 0.5 + 2.0 * np.abs(X[:, 0])
    y = 1.0 + 2.0 * X[:, 0] + rng.standard_normal(n) * sigma
    return y, X


def test_hc_types(het_data):
    y, X = het_data
    for hc in ["HC0", "HC1", "HC2", "HC3"]:
        res = hc_robust_se(y, X, hc_type=hc)
        assert res.extra["hc_type"] == hc
        assert all(np.isfinite(s) for s in res.se.values())


def test_coefficients_same_across_hc(het_data):
    y, X = het_data
    coefs = {}
    for hc in ["HC0", "HC1", "HC2", "HC3"]:
        res = hc_robust_se(y, X, hc_type=hc)
        coefs[hc] = list(res.coefficients.values())
    for hc in ["HC1", "HC2", "HC3"]:
        np.testing.assert_allclose(coefs["HC0"], coefs[hc], atol=1e-10)


def test_hc3_se_largest(het_data):
    y, X = het_data
    se = {}
    for hc in ["HC0", "HC1", "HC2", "HC3"]:
        res = hc_robust_se(y, X, hc_type=hc)
        se[hc] = res.se["x0"]
    assert se["HC3"] >= se["HC2"] - 1e-10
    assert se["HC1"] >= se["HC0"] - 1e-10


def test_invalid_hc_raises():
    with pytest.raises(ValueError, match="HC0"):
        hc_robust_se(np.ones(5), np.ones((5, 1)), hc_type="HC4")


def test_hc1_matches_ols_se_homoskedastic():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 1))
    y = 1.0 + 2.0 * X[:, 0] + rng.standard_normal(n) * 0.5
    from morie.fn.olsrg import ols_regression
    ols = ols_regression(y, X)
    hc1 = hc_robust_se(y, X, hc_type="HC1")
    np.testing.assert_allclose(ols.se["x0"], hc1.se["x0"], rtol=0.15)
