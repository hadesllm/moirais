"""Tests for bhatt.py - Bhattacharyya divergence."""
import numpy as np
import pytest
from morie.fn.bhatt import bhatt_fn, bhatt


def test_bhatt_returns_descriptive_result():
    rng = np.random.default_rng(42)
    X1 = rng.standard_normal((30, 3))
    X2 = rng.standard_normal((30, 3)) + 3.0
    result = bhatt_fn(X1, X2)
    assert result.name == "bhattacharyya"
    assert "divergence" in result.extra


def test_bhatt_identical_near_zero():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 2))
    result = bhatt_fn(X, X)
    assert result.value < 0.5


def test_bhatt_separated_larger():
    rng = np.random.default_rng(42)
    X1 = rng.standard_normal((50, 2))
    X2 = rng.standard_normal((50, 2)) + 10.0
    result = bhatt_fn(X1, X2)
    assert result.value > 1.0


def test_bhatt_alias():
    rng = np.random.default_rng(42)
    X1 = rng.standard_normal((20, 2))
    X2 = rng.standard_normal((20, 2)) + 2.0
    result = bhatt(X1, X2)
    assert result.name == "bhattacharyya"
