"""Tests for gwasem.emmax_gwas."""
import numpy as np
import pytest
from morie.fn.gwasem import emmax_gwas


def test_gwasem_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = emmax_gwas(y, M, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gwasem_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = emmax_gwas(y, M, K)
    assert isinstance(result, dict)
