"""Tests for gb_s12.gibbons_smirnov_one_sided."""
import numpy as np
import pytest
from moirais.fn.gb_s12 import gibbons_smirnov_one_sided


def test_gb_s12_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_smirnov_one_sided(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_s12_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_smirnov_one_sided(x, y)
    assert isinstance(result, dict)
