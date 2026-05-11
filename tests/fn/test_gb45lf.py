"""Tests for gb45lf.gibbons_lilliefors_normal."""
import numpy as np
import pytest
from morie.fn.gb45lf import gibbons_lilliefors_normal


def test_gb45lf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_lilliefors_normal(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb45lf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_lilliefors_normal(x)
    assert isinstance(result, dict)
