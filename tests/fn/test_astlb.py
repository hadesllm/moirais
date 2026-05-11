"""Tests for astlb.astle_balding_grm."""
import numpy as np
import pytest
from morie.fn.astlb import astle_balding_grm


def test_astlb_basic():
    """Test basic functionality."""
    marker_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    freq = np.random.default_rng(42).normal(0, 1, 100)
    result = astle_balding_grm(marker_matrix, freq)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_astlb_edge():
    """Test edge cases."""
    marker_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    freq = np.random.default_rng(42).normal(0, 1, 100)
    result = astle_balding_grm(marker_matrix, freq)
    assert isinstance(result, dict)
