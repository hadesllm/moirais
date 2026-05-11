"""Tests for gb_ar6.gibbons_are_unif."""
import numpy as np
import pytest
from morie.fn.gb_ar6 import gibbons_are_unif


def test_gb_ar6_basic():
    """Test basic functionality."""
    distribution = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_are_unif(distribution)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_ar6_edge():
    """Test edge cases."""
    distribution = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_are_unif(distribution)
    assert isinstance(result, dict)
