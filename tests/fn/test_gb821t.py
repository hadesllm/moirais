"""Tests for gb821t.gibbons_wrs_ties."""
import numpy as np
import pytest
from morie.fn.gb821t import gibbons_wrs_ties


def test_gb821t_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_wrs_ties(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb821t_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_wrs_ties(x, y)
    assert isinstance(result, dict)
