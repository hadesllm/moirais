"""Tests for ncfs.py - Neighbourhood Component Feature Selection."""
import numpy as np
import pytest
from moirais.fn.ncfs import ncfs_fn, ncfs


def test_ncfs_returns_descriptive_result():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 10))
    y = np.array([0]*15 + [1]*15)
    result = ncfs_fn(X, y, n_features=3)
    assert result.name == "ncfs"
    assert result.value == 3
    assert "selected_features" in result.extra


def test_ncfs_correct_count():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 8))
    y = np.array([0]*10 + [1]*10)
    result = ncfs_fn(X, y, n_features=5)
    assert len(result.extra["selected_features"]) == 5


def test_ncfs_unique_features():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 6))
    y = np.array([0]*10 + [1]*10)
    result = ncfs_fn(X, y, n_features=4)
    sel = result.extra["selected_features"]
    assert len(set(sel)) == len(sel)


def test_ncfs_alias():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 5))
    y = np.array([0]*10 + [1]*10)
    result = ncfs(X, y, n_features=2)
    assert result.name == "ncfs"
