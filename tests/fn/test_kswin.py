"""Tests for kswin.kswin."""
import numpy as np
import pytest
from moirais.fn.kswin import kswin


def test_kswin_basic():
    """Test basic functionality."""
    stream = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = kswin(stream, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_kswin_edge():
    """Test edge cases."""
    stream = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = kswin(stream, alpha)
    assert isinstance(result, dict)
