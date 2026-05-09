"""Tests for dctln.py - Dictionary Learning."""
import numpy as np
import pytest
from moirais.fn.dctln import dctln_fn, dctln


def test_dctln_returns_descriptive_result():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((32, 20))
    result = dctln_fn(X, n_atoms=5, n_iter=5)
    assert result.name == "dictionary_learning"
    assert result.value == 5
    assert "dictionary" in result.extra
    assert "codes" in result.extra


def test_dctln_dictionary_shape():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((32, 20))
    result = dctln_fn(X, n_atoms=5, n_iter=5)
    assert result.extra["dictionary"].shape == (32, 5)
    assert result.extra["codes"].shape == (5, 20)


def test_dctln_atoms_normalized():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((16, 10))
    result = dctln_fn(X, n_atoms=3, n_iter=10)
    D = result.extra["dictionary"]
    for k in range(D.shape[1]):
        assert abs(np.linalg.norm(D[:, k]) - 1.0) < 0.1


def test_dctln_alias():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((16, 10))
    result = dctln(X, n_atoms=3, n_iter=3)
    assert result.name == "dictionary_learning"
