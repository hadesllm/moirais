"""Tests for sgtrnh.sgt_randic_index."""
import numpy as np
import pytest
from morie.fn.sgtrnh import sgt_randic_index


def test_sgtrnh_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_randic_index(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtrnh_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_randic_index(A)
    assert isinstance(result, dict)
