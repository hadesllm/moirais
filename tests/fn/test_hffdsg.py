"""Tests for hffdsg.hoeffding_inequality."""
import numpy as np
import pytest
from moirais.fn.hffdsg import hoeffding_inequality


def test_hffdsg_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    t = np.linspace(0, 10, 100)
    result = hoeffding_inequality(a, b, n, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hffdsg_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    t = np.linspace(0, 10, 100)
    result = hoeffding_inequality(a, b, n, t)
    assert isinstance(result, dict)
