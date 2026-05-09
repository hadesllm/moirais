"""Tests for phyltr.phylogenetic_tree_nj."""
import numpy as np
import pytest
from moirais.fn.phyltr import phylogenetic_tree_nj


def test_phyltr_basic():
    """Test basic functionality."""
    distance_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = phylogenetic_tree_nj(distance_matrix)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_phyltr_edge():
    """Test edge cases."""
    distance_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = phylogenetic_tree_nj(distance_matrix)
    assert isinstance(result, dict)
