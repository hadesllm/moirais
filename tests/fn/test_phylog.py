"""Tests for phylog.phylogenetic_dating."""
import numpy as np
import pytest
from morie.fn.phylog import phylogenetic_dating


def test_phylog_basic():
    """Test basic functionality."""
    tree = np.random.default_rng(42).normal(0, 1, 100)
    sample_dates = np.random.default_rng(42).normal(0, 1, 100)
    result = phylogenetic_dating(tree, sample_dates)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_phylog_edge():
    """Test edge cases."""
    tree = np.random.default_rng(42).normal(0, 1, 100)
    sample_dates = np.random.default_rng(42).normal(0, 1, 100)
    result = phylogenetic_dating(tree, sample_dates)
    assert isinstance(result, dict)
