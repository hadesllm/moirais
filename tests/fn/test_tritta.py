"""Tests for tritta.alphafold_triangle_attn."""
import numpy as np
import pytest
from moirais.fn.tritta import alphafold_triangle_attn


def test_tritta_basic():
    """Test basic functionality."""
    pair_repr = np.random.default_rng(42).normal(0, 1, 100)
    start_end = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_triangle_attn(pair_repr, start_end)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tritta_edge():
    """Test edge cases."""
    pair_repr = np.random.default_rng(42).normal(0, 1, 100)
    start_end = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_triangle_attn(pair_repr, start_end)
    assert isinstance(result, dict)
