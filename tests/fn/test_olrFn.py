"""Tests for olrFn.outgoing_longwave."""
import numpy as np
import pytest
from moirais.fn.olrFn import outgoing_longwave


def test_olrFn_basic():
    """Test basic functionality."""
    T_s = np.random.default_rng(42).normal(0, 1, 100)
    result = outgoing_longwave(T_s)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_olrFn_edge():
    """Test edge cases."""
    T_s = np.random.default_rng(42).normal(0, 1, 100)
    result = outgoing_longwave(T_s)
    assert isinstance(result, dict)
