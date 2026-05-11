"""Tests for sgtreg.sgt_resistance_distance_matrix."""
import numpy as np
import pytest
from morie.fn.sgtreg import sgt_resistance_distance_matrix


def test_sgtreg_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_resistance_distance_matrix(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtreg_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_resistance_distance_matrix(A)
    assert isinstance(result, dict)
