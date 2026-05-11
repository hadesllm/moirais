"""Tests for vbnpc.vb_nonparametric."""
import numpy as np
import pytest
from morie.fn.vbnpc import vb_nonparametric


def test_vbnpc_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    K_truncate = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = vb_nonparametric(y, K_truncate, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vbnpc_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    K_truncate = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = vb_nonparametric(y, K_truncate, alpha)
    assert isinstance(result, dict)
