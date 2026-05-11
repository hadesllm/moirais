"""Tests for morie.fn.gee — Generalized estimating equations."""

import numpy as np
import pytest

from morie.fn.gee import gee_regression


def test_gee_gaussian_recovers_coef():
    rng = np.random.default_rng(42)
    n_clusters = 50
    cluster_size = 10
    y_list, x_list, c_list = [], [], []
    for c in range(n_clusters):
        u = rng.standard_normal()
        for _ in range(cluster_size):
            x = rng.standard_normal()
            y = 1.0 + 2.0 * x + u + rng.standard_normal() * 0.5
            y_list.append(y)
            x_list.append([x])
            c_list.append(c)
    y = np.array(y_list)
    X = np.array(x_list)
    clusters = np.array(c_list)
    res = gee_regression(y, X, clusters)
    assert abs(res.coefficients["x0"] - 2.0) < 0.5


def test_gee_binomial():
    rng = np.random.default_rng(42)
    y, X, cl = [], [], []
    for c in range(30):
        for _ in range(5):
            x = rng.standard_normal()
            p = 1 / (1 + np.exp(-(1.0 + x)))
            y.append(float(rng.uniform() < p))
            X.append([x])
            cl.append(c)
    res = gee_regression(np.array(y), np.array(X), np.array(cl),
                         family="binomial")
    assert res.coefficients["x0"] > 0


def test_gee_exchangeable():
    rng = np.random.default_rng(7)
    y, X, cl = [], [], []
    for c in range(40):
        u = rng.standard_normal()
        for _ in range(8):
            x = rng.standard_normal()
            y.append(1.0 + x + u + rng.standard_normal() * 0.3)
            X.append([x])
            cl.append(c)
    res = gee_regression(np.array(y), np.array(X), np.array(cl),
                         corr_structure="exchangeable")
    assert res.extra["n_clusters"] == 40
