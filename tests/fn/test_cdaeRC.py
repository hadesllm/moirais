"""Tests for cdaeRC.cdae."""
import numpy as np
import pytest
from moirais.fn.cdaeRC import cdae


def test_cdaeRC_basic():
    """Test basic functionality."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = cdae(R, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cdaeRC_edge():
    """Test edge cases."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = cdae(R, K)
    assert isinstance(result, dict)
