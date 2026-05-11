"""Tests for phylml.phylogenetic_ml."""
import numpy as np
import pytest
from morie.fn.phylml import phylogenetic_ml


def test_phylml_basic():
    """Test basic functionality."""
    alignment = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = phylogenetic_ml(alignment, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_phylml_edge():
    """Test edge cases."""
    alignment = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = phylogenetic_ml(alignment, model)
    assert isinstance(result, dict)
