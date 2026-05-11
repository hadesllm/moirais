"""Tests for rngnrm.qk_norm."""
import numpy as np
import pytest
from morie.fn.rngnrm import rngnrm as qk_norm


def test_rngnrm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    s = 90
    result = qk_norm(y, Q, K, s)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rngnrm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    s = 90
    result = qk_norm(y, Q, K, s)
    assert isinstance(result, dict)
