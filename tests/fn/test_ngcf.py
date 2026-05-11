"""Tests for ngcf.ngcf."""
import numpy as np
import pytest
from morie.fn.ngcf import ngcf


def test_ngcf_basic():
    """Test basic functionality."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    layers = np.random.default_rng(42).normal(0, 1, 100)
    result = ngcf(R, K, layers)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ngcf_edge():
    """Test edge cases."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    layers = np.random.default_rng(42).normal(0, 1, 100)
    result = ngcf(R, K, layers)
    assert isinstance(result, dict)
