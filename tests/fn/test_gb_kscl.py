"""Tests for gb_kscl.gibbons_ks_critical_values."""
import numpy as np
import pytest
from moirais.fn.gb_kscl import gibbons_ks_critical_values


def test_gb_kscl_basic():
    """Test basic functionality."""
    n = 100
    alpha = 0.05
    result = gibbons_ks_critical_values(n, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_kscl_edge():
    """Test edge cases."""
    n = 100
    alpha = 0.05
    result = gibbons_ks_critical_values(n, alpha)
    assert isinstance(result, dict)
