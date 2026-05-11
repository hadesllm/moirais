"""Tests for morie.fn.clogr — Conditional logistic regression."""

import numpy as np
import pytest

from morie.fn.clogr import conditional_logistic


def test_clogr_positive_coef():
    rng = np.random.default_rng(42)
    n_strata = 100
    y_list, x_list, s_list = [], [], []
    for s in range(n_strata):
        x_case = rng.standard_normal(2) + 1.0
        x_ctrl = rng.standard_normal(2)
        y_list.extend([1, 0])
        x_list.append(x_case)
        x_list.append(x_ctrl)
        s_list.extend([s, s])
    y = np.array(y_list)
    X = np.array(x_list)
    strata = np.array(s_list)
    res = conditional_logistic(y, X, strata)
    assert res.coefficients["x0"] > 0


def test_clogr_no_informative_strata_raises():
    y = np.array([1, 1, 0, 0])
    X = np.ones((4, 1))
    strata = np.array([0, 1, 2, 3])
    with pytest.raises(ValueError, match="informative"):
        conditional_logistic(y, X, strata)


def test_clogr_n_strata():
    rng = np.random.default_rng(7)
    n_s = 50
    y, X, s = [], [], []
    for i in range(n_s):
        y.extend([1, 0])
        X.append(rng.standard_normal(1))
        X.append(rng.standard_normal(1))
        s.extend([i, i])
    res = conditional_logistic(np.array(y), np.array(X), np.array(s))
    assert res.extra["n_strata"] == n_s
