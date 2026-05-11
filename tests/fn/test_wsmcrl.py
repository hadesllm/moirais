"""Tests for wsmcrl.wasserman_cramer_rao."""
import numpy as np
import pytest
from morie.fn.wsmcrl import wasserman_cramer_rao


def test_wsmcrl_basic():
    """Test basic functionality."""
    theta = 0.0
    n = 100
    I = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_cramer_rao(theta, n, I)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmcrl_edge():
    """Test edge cases."""
    theta = 0.0
    n = 100
    I = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_cramer_rao(theta, n, I)
    assert isinstance(result, dict)
