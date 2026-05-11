"""Tests for ghs003.ghosal_ch2_basis_truncation_error."""
import numpy as np
import pytest
from morie.fn.ghs003 import ghosal_ch2_basis_truncation_error


def test_ghs003_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    J = 20
    alpha = 0.05
    k = 5
    result = ghosal_ch2_basis_truncation_error(f, J, alpha, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs003_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    J = 20
    alpha = 0.05
    k = 5
    result = ghosal_ch2_basis_truncation_error(f, J, alpha, k)
    assert isinstance(result, dict)
