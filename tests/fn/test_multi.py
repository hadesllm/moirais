"""Tests for morie.fn.multi — Multinomial logit."""

import numpy as np
import pytest

from morie.fn.multi import multinomial_logit


def test_multinomial_three_classes():
    rng = np.random.default_rng(42)
    n = 600
    X = rng.standard_normal((n, 2))
    scores = np.column_stack([
        np.zeros(n),
        1.0 + X[:, 0],
        -1.0 + X[:, 1],
    ])
    probs = np.exp(scores)
    probs /= probs.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=p) for p in probs])
    res = multinomial_logit(y, X)
    assert res.extra["accuracy"] > 0.2
    assert len(res.extra["categories"]) == 3


def test_multinomial_binary_raises():
    y = np.array([0, 0, 1, 1, 0])
    X = np.ones((5, 1))
    with pytest.raises(ValueError, match="binary"):
        multinomial_logit(y, X)


def test_multinomial_aic_finite():
    rng = np.random.default_rng(7)
    n = 200
    X = rng.standard_normal((n, 2))
    y = rng.choice(4, size=n)
    res = multinomial_logit(y, X)
    assert np.isfinite(res.extra["aic"])
