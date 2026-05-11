"""Tests for vaeCF.vae_cf."""
import numpy as np
import pytest
from morie.fn.vaeCF import vae_cf


def test_vaeCF_basic():
    """Test basic functionality."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = vae_cf(R, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vaeCF_edge():
    """Test edge cases."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = vae_cf(R, K)
    assert isinstance(result, dict)
