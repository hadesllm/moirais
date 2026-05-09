"""Tests for sgnpw.sign_test_power."""
import numpy as np
import pytest
from moirais.fn.sgnpw import sign_test_power


def test_sgnpw_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = sign_test_power(x)
    assert 'statistic' in result
    assert 'p_value' in result
    assert 0 <= result['p_value'] <= 1


def test_sgnpw_edge():
    """Test edge cases."""
    result = sign_test_power(np.array([1.0]))
    assert result['n'] == 1
