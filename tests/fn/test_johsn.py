"""Tests for johsn.johansen_cointegration."""
import numpy as np
import pytest
from morie.fn.johsn import johansen_cointegration


def test_johsn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = johansen_cointegration(x)
    assert 'statistic' in result
    assert 'p_value' in result
    assert 0 <= result['p_value'] <= 1


def test_johsn_edge():
    """Test edge cases."""
    result = johansen_cointegration(np.array([1.0]))
    assert result['n'] == 1
