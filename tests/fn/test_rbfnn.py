"""Tests for rbfnn.py - RBF network."""
import numpy as np
import pytest
from morie.fn.rbfnn import rbfnn_fn, rbfnn


def test_rbfnn_returns_descriptive_result():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 3))
    y = rng.standard_normal(50)
    result = rbfnn_fn(X, y, n_centers=5)
    assert result.name == "rbf_network"
    assert "weights" in result.extra
    assert "centers" in result.extra
    assert "sigma" in result.extra


def test_rbfnn_center_count():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 3))
    y = rng.standard_normal(50)
    result = rbfnn_fn(X, y, n_centers=5)
    assert result.extra["centers"].shape[0] == 5


def test_rbfnn_weight_count():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 3))
    y = rng.standard_normal(50)
    result = rbfnn_fn(X, y, n_centers=8)
    assert result.value == 8


def test_rbfnn_alias():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 2))
    y = rng.standard_normal(30)
    result = rbfnn(X, y, n_centers=3)
    assert result.name == "rbf_network"
