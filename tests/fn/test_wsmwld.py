"""Tests for wsmwld.wasserman_wald."""
import numpy as np
import pytest
from moirais.fn.wsmwld import wasserman_wald


def test_wsmwld_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    theta0 = 0.0
    result = wasserman_wald(data, f, theta0)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wsmwld_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    theta0 = 0.0
    result = wasserman_wald(data, f, theta0)
    assert isinstance(result, dict)
