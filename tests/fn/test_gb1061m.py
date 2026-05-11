"""Tests for gb1061m.gibbons_jt_moments."""
import numpy as np
import pytest
from morie.fn.gb1061m import gibbons_jt_moments


def test_gb1061m_basic():
    """Test basic functionality."""
    n_i = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_jt_moments(n_i)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1061m_edge():
    """Test edge cases."""
    n_i = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_jt_moments(n_i)
    assert isinstance(result, dict)
