"""Tests for rcov.py - Riemannian covariance."""
import numpy as np
import pytest
from morie.fn.rcov import rcov_fn, rcov


def test_rcov_returns_descriptive_result():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 100))
    result = rcov_fn(X)
    assert result.name == "riemannian_cov"
    assert "covariance" in result.extra
    assert "metric" in result.extra


def test_rcov_symmetric():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 100))
    result = rcov_fn(X)
    C = result.extra["covariance"]
    assert np.allclose(C, C.T)


def test_rcov_logeuclid():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 100))
    result = rcov_fn(X, metric="logeuclid")
    assert result.extra["metric"] == "logeuclid"
    C = result.extra["covariance"]
    assert np.allclose(C, C.T)


def test_rcov_alias():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((2, 50))
    result = rcov(X)
    assert result.name == "riemannian_cov"
