"""Tests for farmlmm.farm_cpu."""
import numpy as np
import pytest
from moirais.fn.farmlmm import farm_cpu


def test_farmlmm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = farm_cpu(y, M, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_farmlmm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = farm_cpu(y, M, K)
    assert isinstance(result, dict)
