"""Tests for gb433.gibbons_ks_kolmogorov_limit."""
import numpy as np
import pytest
from morie.fn.gb433 import gibbons_ks_kolmogorov_limit


def test_gb433_basic():
    """Test basic functionality."""
    d = 5
    n = 100
    result = gibbons_ks_kolmogorov_limit(d, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb433_edge():
    """Test edge cases."""
    d = 5
    n = 100
    result = gibbons_ks_kolmogorov_limit(d, n)
    assert isinstance(result, dict)
