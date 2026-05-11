"""Tests for phylotr.phylogenetic_tree."""
import numpy as np
import pytest
from morie.fn.phylotr import phylogenetic_tree


def test_phylotr_basic():
    """Test basic functionality."""
    sequences = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = phylogenetic_tree(sequences, method)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_phylotr_edge():
    """Test edge cases."""
    sequences = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = phylogenetic_tree(sequences, method)
    assert isinstance(result, dict)
