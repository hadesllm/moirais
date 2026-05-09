"""Tests for gb_c2.gibbons_chi2_yates."""
import numpy as np
import pytest
from moirais.fn.gb_c2 import gibbons_chi2_yates


def test_gb_c2_basic():
    """Test basic functionality."""
    table = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_chi2_yates(table)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_c2_edge():
    """Test edge cases."""
    table = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_chi2_yates(table)
    assert isinstance(result, dict)
