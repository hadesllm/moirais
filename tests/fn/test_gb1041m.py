"""Tests for gb1041m.gibbons_kw_mult_comp."""
import numpy as np
import pytest
from morie.fn.gb1041m import gibbons_kw_mult_comp


def test_gb1041m_basic():
    """Test basic functionality."""
    groups = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_kw_mult_comp(groups, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1041m_edge():
    """Test edge cases."""
    groups = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = gibbons_kw_mult_comp(groups, alpha)
    assert isinstance(result, dict)
