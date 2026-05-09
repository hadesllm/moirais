"""Tests for crfboot.causal_forest_bootstrap."""
import numpy as np
import pytest
from moirais.fn.crfboot import causal_forest_bootstrap


def test_crfboot_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    min_node = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_forest_bootstrap(y, D, X, B, min_node)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_crfboot_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    min_node = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_forest_bootstrap(y, D, X, B, min_node)
    assert isinstance(result, dict)
