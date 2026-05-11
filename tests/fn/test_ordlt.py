"""Tests for morie.fn.ordlt — Ordered logit."""

import numpy as np
import pytest

from morie.fn.ordlt import ordered_logit


def test_ordered_logit_cutpoints_increasing():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 1))
    latent = 1.0 * X[:, 0] + rng.logistic(size=n)
    y = np.digitize(latent, bins=[-1, 0, 1])
    res = ordered_logit(y, X)
    cuts = res.extra["cutpoints"]
    for i in range(len(cuts) - 1):
        assert cuts[i] < cuts[i + 1]


def test_ordered_logit_positive_coef():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 1))
    latent = 2.0 * X[:, 0] + rng.logistic(size=n)
    y = np.digitize(latent, bins=[-1, 0, 1])
    res = ordered_logit(y, X)
    assert res.coefficients["x0"] > 0


def test_ordered_logit_binary_raises():
    y = np.array([0, 0, 1, 1])
    X = np.ones((4, 1))
    with pytest.raises(ValueError, match="3 ordered"):
        ordered_logit(y, X)


def test_ordered_logit_accuracy():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 1))
    latent = 2.0 * X[:, 0] + rng.logistic(size=n)
    y = np.digitize(latent, bins=[-1, 0, 1])
    res = ordered_logit(y, X)
    assert res.extra["accuracy"] > 0.3
