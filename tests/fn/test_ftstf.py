"""Tests for f_test_features."""
import numpy as np
import pytest
from moirais.fn.ftstf import f_test_features, ftstf


def test_basic():
    rng = np.random.default_rng(42)
    X = np.column_stack([rng.normal(0, 1, 100), rng.normal(0, 0.01, 100)])
    y = np.array([0]*50 + [1]*50)
    X[:50, 0] += 5
    r = f_test_features(X, y)
    assert r.extra["ranking"][0] == 0


def test_alias():
    assert ftstf is f_test_features


def test_single_class():
    with pytest.raises(ValueError):
        f_test_features([[1], [2]], [0, 0])
