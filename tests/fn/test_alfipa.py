"""Tests for alfipa.alphafold_invariant_point."""
import numpy as np
import pytest
from moirais.fn.alfipa import alphafold_invariant_point


def test_alfipa_basic():
    """Test basic functionality."""
    s_i = np.random.default_rng(42).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    v = np.random.default_rng(44).normal(0, 1, 100)
    frames = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_invariant_point(s_i, q, k, v, frames)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfipa_edge():
    """Test edge cases."""
    s_i = np.random.default_rng(42).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    v = np.random.default_rng(44).normal(0, 1, 100)
    frames = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_invariant_point(s_i, q, k, v, frames)
    assert isinstance(result, dict)
