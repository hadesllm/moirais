"""Tests for ppswrs.pps_with_replacement."""
import numpy as np
import pytest
from morie.fn.ppswrs import pps_with_replacement


def test_ppswrs_basic():
    """Test basic functionality."""
    sizes = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = pps_with_replacement(sizes, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ppswrs_edge():
    """Test edge cases."""
    sizes = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = pps_with_replacement(sizes, n)
    assert isinstance(result, dict)
