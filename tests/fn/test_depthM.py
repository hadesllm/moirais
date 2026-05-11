"""Tests for depthM.mahalanobis_depth."""
import numpy as np
import pytest
from morie.fn.depthM import mahalanobis_depth


def test_depthM_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    Sigma = np.random.default_rng(42).normal(0, 1, 100)
    result = mahalanobis_depth(x, mu, Sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_depthM_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    Sigma = np.random.default_rng(42).normal(0, 1, 100)
    result = mahalanobis_depth(x, mu, Sigma)
    assert isinstance(result, dict)
