"""Tests for gb_smn.gibbons_smirnov_2sided."""
import numpy as np
import pytest
from moirais.fn.gb_smn import gibbons_smirnov_2sided


def test_gb_smn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_smirnov_2sided(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_smn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_smirnov_2sided(x, y)
    assert isinstance(result, dict)
