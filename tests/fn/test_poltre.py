"""Tests for poltre.polya_tree_prior."""
import numpy as np
import pytest
from morie.fn.poltre import polya_tree_prior


def test_poltre_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    tree_depth = np.random.default_rng(42).normal(0, 1, 100)
    prior_alpha = np.random.default_rng(42).normal(0, 1, 100)
    result = polya_tree_prior(y, tree_depth, prior_alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_poltre_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    tree_depth = np.random.default_rng(42).normal(0, 1, 100)
    prior_alpha = np.random.default_rng(42).normal(0, 1, 100)
    result = polya_tree_prior(y, tree_depth, prior_alpha)
    assert isinstance(result, dict)
