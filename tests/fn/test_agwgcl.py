"""Tests for agwgcl.alphazero_weight_clipping."""
import numpy as np
import pytest
from morie.fn.agwgcl import alphazero_weight_clipping


def test_agwgcl_basic():
    """Test basic functionality."""
    grad = np.random.default_rng(42).normal(0, 1, 100)
    max_norm = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_weight_clipping(grad, max_norm)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agwgcl_edge():
    """Test edge cases."""
    grad = np.random.default_rng(42).normal(0, 1, 100)
    max_norm = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_weight_clipping(grad, max_norm)
    assert isinstance(result, dict)
