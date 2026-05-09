"""Tests for bkcbow.burkov_cbow."""
import numpy as np
import pytest
from moirais.fn.bkcbow import burkov_cbow


def test_bkcbow_basic():
    """Test basic functionality."""
    context_indices = np.random.default_rng(42).normal(0, 1, 100)
    center_index = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    U = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_cbow(context_indices, center_index, V, U)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bkcbow_edge():
    """Test edge cases."""
    context_indices = np.random.default_rng(42).normal(0, 1, 100)
    center_index = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    U = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_cbow(context_indices, center_index, V, U)
    assert isinstance(result, dict)
