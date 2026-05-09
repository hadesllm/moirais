"""Tests for sgtwh.sgt_wiener_index."""
import numpy as np
import pytest
from moirais.fn.sgtwh import sgt_wiener_index


def test_sgtwh_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_wiener_index(A)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_sgtwh_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_wiener_index(A)
    assert isinstance(result, dict)
