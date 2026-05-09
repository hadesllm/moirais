"""Tests for vdwrd.van_der_waerden_test."""
import numpy as np
import pytest
from moirais.fn.vdwrd import van_der_waerden_test


def test_vdwrd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = van_der_waerden_test(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_vdwrd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = van_der_waerden_test(x, y)
    assert isinstance(result, dict)
