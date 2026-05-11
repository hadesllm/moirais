"""Tests for gb5712.gibbons_wsrt_power."""
import numpy as np
import pytest
from morie.fn.gb5712 import gibbons_wsrt_power


def test_gb5712_basic():
    """Test basic functionality."""
    theta = 0.0
    n = 100
    alpha = 0.05
    result = gibbons_wsrt_power(theta, n, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb5712_edge():
    """Test edge cases."""
    theta = 0.0
    n = 100
    alpha = 0.05
    result = gibbons_wsrt_power(theta, n, alpha)
    assert isinstance(result, dict)
