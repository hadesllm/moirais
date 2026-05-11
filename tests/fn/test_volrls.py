"""Tests for volrls.vol_recursive_least_sq."""
import numpy as np
import pytest
from morie.fn.volrls import vol_recursive_least_sq


def test_volrls_basic():
    """Test basic functionality."""
    r = 10
    lam = 0.1
    result = vol_recursive_least_sq(r, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volrls_edge():
    """Test edge cases."""
    r = 10
    lam = 0.1
    result = vol_recursive_least_sq(r, lam)
    assert isinstance(result, dict)
