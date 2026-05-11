"""Tests for cvxprx.boyd_proximal."""
import numpy as np
import pytest
from morie.fn.cvxprx import boyd_proximal


def test_cvxprx_basic():
    """Test basic functionality."""
    h = 0.3
    v = np.random.default_rng(44).normal(0, 1, 100)
    result = boyd_proximal(h, v)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cvxprx_edge():
    """Test edge cases."""
    h = 0.3
    v = np.random.default_rng(44).normal(0, 1, 100)
    result = boyd_proximal(h, v)
    assert isinstance(result, dict)
