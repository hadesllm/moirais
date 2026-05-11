"""Tests for cuttip.cutting_plane."""
import numpy as np
import pytest
from morie.fn.cuttip import cutting_plane


def test_cuttip_basic():
    """Test basic functionality."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    integer_indices = np.random.default_rng(42).normal(0, 1, 100)
    result = cutting_plane(c, A, b, integer_indices)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cuttip_edge():
    """Test edge cases."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    integer_indices = np.random.default_rng(42).normal(0, 1, 100)
    result = cutting_plane(c, A, b, integer_indices)
    assert isinstance(result, dict)
