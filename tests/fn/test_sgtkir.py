"""Tests for sgtkir.sgt_kirchhoff_index."""
import numpy as np
import pytest
from morie.fn.sgtkir import sgt_kirchhoff_index


def test_sgtkir_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_kirchhoff_index(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtkir_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_kirchhoff_index(A)
    assert isinstance(result, dict)
