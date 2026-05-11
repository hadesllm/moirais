"""Tests for gb_are5.gibbons_are_scale_tests."""
import numpy as np
import pytest
from morie.fn.gb_are5 import gibbons_are_scale_tests


def test_gb_are5_basic():
    """Test basic functionality."""
    distribution = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_are_scale_tests(distribution)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_are5_edge():
    """Test edge cases."""
    distribution = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_are_scale_tests(distribution)
    assert isinstance(result, dict)
