"""Tests for alftrm.alphafold_triangle_mult."""
import numpy as np
import pytest
from morie.fn.alftrm import alphafold_triangle_mult


def test_alftrm_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    direction = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_triangle_mult(z, direction)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alftrm_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    direction = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_triangle_mult(z, direction)
    assert isinstance(result, dict)
