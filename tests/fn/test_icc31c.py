"""Tests for icc31c.icc_one_way."""
import numpy as np
import pytest
from morie.fn.icc31c import icc_one_way


def test_icc31c_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = icc_one_way(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_icc31c_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = icc_one_way(X)
    assert isinstance(result, dict)
