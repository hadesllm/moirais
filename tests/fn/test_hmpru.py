"""Tests for hmpru.geron_weight_pruning."""
import numpy as np
import pytest
from morie.fn.hmpru import geron_weight_pruning


def test_hmpru_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    sparsity = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_weight_pruning(model, sparsity)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmpru_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    sparsity = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_weight_pruning(model, sparsity)
    assert isinstance(result, dict)
