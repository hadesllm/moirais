"""Tests for egarcm.egarch_nelson."""
import numpy as np
import pytest
from morie.fn.egarcm import egarch_nelson


def test_egarcm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = egarch_nelson(x, p, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_egarcm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = egarch_nelson(x, p, q)
    assert isinstance(result, dict)
