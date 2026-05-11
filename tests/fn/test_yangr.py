"""Tests for yangr.yang_realized_relationship."""
import numpy as np
import pytest
from morie.fn.yangr import yang_realized_relationship


def test_yangr_basic():
    """Test basic functionality."""
    marker_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    freq = np.random.default_rng(42).normal(0, 1, 100)
    result = yang_realized_relationship(marker_matrix, freq)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_yangr_edge():
    """Test edge cases."""
    marker_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    freq = np.random.default_rng(42).normal(0, 1, 100)
    result = yang_realized_relationship(marker_matrix, freq)
    assert isinstance(result, dict)
