"""Tests for morie.fn.whitt — White's test."""

import numpy as np
import pytest

from morie.fn.whitt import white_test


def test_white_detects_heteroskedasticity():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 1))
    sigma = 0.5 + 2.0 * np.abs(X[:, 0])
    y = 1.0 + X[:, 0] + rng.standard_normal(n) * sigma
    res = white_test(y, X)
    assert res.extra["reject_H0"] is True


def test_white_homoskedastic():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 1))
    y = 1.0 + X[:, 0] + rng.standard_normal(n) * 0.5
    res = white_test(y, X)
    assert res.value >= 0
    assert res.extra["p_value"] >= 0


def test_white_df_correct():
    rng = np.random.default_rng(7)
    n = 100
    X = rng.standard_normal((n, 2))
    y = rng.standard_normal(n)
    res = white_test(y, X)
    assert res.extra["df"] == 5


def test_white_single_predictor():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 1))
    y = rng.standard_normal(n)
    res = white_test(y, X)
    assert res.extra["df"] == 2
