"""Tests for moirais.fn.panel — Panel data regression."""

import numpy as np
import pytest

from moirais.fn.panel import panel_regression


@pytest.fixture()
def panel_data():
    rng = np.random.default_rng(42)
    N, T = 50, 10
    y_list, x_list, e_list = [], [], []
    for i in range(N):
        alpha_i = rng.standard_normal()
        for t in range(T):
            x = rng.standard_normal()
            y = alpha_i + 2.0 * x + rng.standard_normal() * 0.5
            y_list.append(y)
            x_list.append([x])
            e_list.append(i)
    return np.array(y_list), np.array(x_list), np.array(e_list)


def test_fe_recovers_coef(panel_data):
    y, X, ent = panel_data
    res = panel_regression(y, X, ent, method="fe")
    assert abs(res.coefficients["x0"] - 2.0) < 0.3


def test_re_recovers_coef(panel_data):
    y, X, ent = panel_data
    res = panel_regression(y, X, ent, method="re")
    assert abs(res.coefficients["x0"] - 2.0) < 0.5


def test_fe_no_intercept(panel_data):
    y, X, ent = panel_data
    res = panel_regression(y, X, ent, method="fe")
    assert "(Intercept)" not in res.coefficients


def test_re_has_intercept(panel_data):
    y, X, ent = panel_data
    res = panel_regression(y, X, ent, method="re")
    assert "(Intercept)" in res.coefficients


def test_invalid_method_raises(panel_data):
    y, X, ent = panel_data
    with pytest.raises(ValueError, match="fe.*re"):
        panel_regression(y, X, ent, method="invalid")
