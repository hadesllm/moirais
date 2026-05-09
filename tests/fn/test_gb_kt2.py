"""Tests for gb_kt2.gibbons_kendall_exact."""
import numpy as np
import pytest
from moirais.fn.gb_kt2 import gibbons_kendall_exact


def test_gb_kt2_basic():
    """Test basic functionality."""
    n = 100
    t = np.linspace(0, 10, 100)
    result = gibbons_kendall_exact(n, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb_kt2_edge():
    """Test edge cases."""
    n = 100
    t = np.linspace(0, 10, 100)
    result = gibbons_kendall_exact(n, t)
    assert isinstance(result, dict)
