"""Tests for grprn.geron_weight_pruning."""
import numpy as np
import pytest
from moirais.fn.grprn import geron_weight_pruning


def test_grprn_basic():
    """Test basic functionality."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    sparsity = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_weight_pruning(W, sparsity)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grprn_edge():
    """Test edge cases."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    sparsity = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_weight_pruning(W, sparsity)
    assert isinstance(result, dict)
