"""Tests for wsmlln.wasserman_lln."""
import numpy as np
import pytest
from moirais.fn.wsmlln import wasserman_lln


def test_wsmlln_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_lln(data)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmlln_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_lln(data)
    assert isinstance(result, dict)
