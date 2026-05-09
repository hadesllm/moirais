"""Tests for likelihood_ratio."""
import numpy as np
import pytest
from moirais.fn.lkrat import likelihood_ratio, lkrat


def test_same_params():
    x = [1.0, 2.0, 3.0]
    r = likelihood_ratio(x, (2.0, 1.0), (2.0, 1.0))
    assert abs(r.estimate - 1.0) < 1e-6


def test_alias():
    assert lkrat is likelihood_ratio


def test_bad_sigma():
    with pytest.raises(ValueError):
        likelihood_ratio([1], (0, 0), (1, 1))


def test_different_hypotheses():
    rng = np.random.default_rng(42)
    x = rng.normal(5.0, 1.0, 100)
    r = likelihood_ratio(x, (0.0, 1.0), (5.0, 1.0))
    assert r.estimate > 1.0
