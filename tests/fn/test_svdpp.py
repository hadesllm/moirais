"""Tests for svdpp.svdpp."""
import numpy as np
import pytest
from morie.fn.svdpp import svdpp


def test_svdpp_basic():
    """Test basic functionality."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    implicit = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = svdpp(R, implicit, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_svdpp_edge():
    """Test edge cases."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    implicit = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = svdpp(R, implicit, K)
    assert isinstance(result, dict)
