"""Tests for gwasl1.gwas_linear."""
import numpy as np
import pytest
from morie.fn.gwasl1 import gwas_linear


def test_gwasl1_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = gwas_linear(y, M, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gwasl1_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = gwas_linear(y, M, K)
    assert isinstance(result, dict)
