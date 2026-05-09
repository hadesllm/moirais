"""Tests for mhlds.py - Mahalanobis distance."""
import numpy as np
import pytest
from moirais.fn.mhlds import mhlds_fn, mhlds


def test_mhlds_returns_descriptive_result():
    x = np.array([1.0, 2.0])
    mu = np.array([0.0, 0.0])
    cov = np.eye(2)
    result = mhlds_fn(x, mu, cov)
    assert result.name == "mahalanobis"
    assert "distance" in result.extra


def test_mhlds_identity_cov_equals_euclidean():
    x = np.array([3.0, 4.0])
    mu = np.zeros(2)
    cov = np.eye(2)
    result = mhlds_fn(x, mu, cov)
    assert abs(result.value - 5.0) < 1e-10


def test_mhlds_zero_at_mean():
    mu = np.array([1.0, 2.0, 3.0])
    cov = np.eye(3) * 2
    result = mhlds_fn(mu, mu, cov)
    assert abs(result.value) < 1e-10


def test_mhlds_alias():
    x = np.array([1.0])
    mu = np.array([0.0])
    cov = np.array([[1.0]])
    result = mhlds(x, mu, cov)
    assert result.name == "mahalanobis"
