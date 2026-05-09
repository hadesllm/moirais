"""Tests for kblup.kernel_blup."""
import numpy as np
import pytest
from moirais.fn.kblup import kernel_blup


def test_kblup_basic():
    """Test basic functionality."""
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    K_new = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    lam = 0.1
    result = kernel_blup(K, K_new, y, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kblup_edge():
    """Test edge cases."""
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    K_new = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    lam = 0.1
    result = kernel_blup(K, K_new, y, lam)
    assert isinstance(result, dict)
