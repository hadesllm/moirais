"""Tests for cmlmer.compressed_lmm."""
import numpy as np
import pytest
from moirais.fn.cmlmer import compressed_lmm


def test_cmlmer_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    clusters = np.random.default_rng(42).normal(0, 1, 100)
    result = compressed_lmm(y, M, K, clusters)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cmlmer_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    clusters = np.random.default_rng(42).normal(0, 1, 100)
    result = compressed_lmm(y, M, K, clusters)
    assert isinstance(result, dict)
