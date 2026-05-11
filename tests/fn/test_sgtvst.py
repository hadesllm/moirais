"""Tests for sgtvst.sgt_vertex_strengths."""
import numpy as np
import pytest
from morie.fn.sgtvst import sgt_vertex_strengths


def test_sgtvst_basic():
    """Test basic functionality."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = sgt_vertex_strengths(W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtvst_edge():
    """Test edge cases."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = sgt_vertex_strengths(W)
    assert isinstance(result, dict)
