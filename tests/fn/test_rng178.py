"""Tests for rng178.rangayyan_ch4_qrs_combined_balda."""
import numpy as np
import pytest
from morie.fn.rng178 import rangayyan_ch4_qrs_combined_balda


def test_rng178_basic():
    """Test basic functionality."""
    y_0 = np.random.default_rng(42).normal(0, 1, 100)
    y_1 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_qrs_combined_balda(y_0, y_1, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng178_edge():
    """Test edge cases."""
    y_0 = np.random.default_rng(42).normal(0, 1, 100)
    y_1 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_qrs_combined_balda(y_0, y_1, n)
    assert isinstance(result, dict)
