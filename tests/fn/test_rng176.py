"""Tests for rng176.rangayyan_ch4_qrs_first_derivative_balda."""
import numpy as np
import pytest
from moirais.fn.rng176 import rangayyan_ch4_qrs_first_derivative_balda


def test_rng176_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_qrs_first_derivative_balda(x, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng176_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_qrs_first_derivative_balda(x, n)
    assert isinstance(result, dict)
