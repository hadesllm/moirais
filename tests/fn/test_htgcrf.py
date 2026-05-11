"""Tests for htgcrf.hetero_causal_forest."""
import numpy as np
import pytest
from morie.fn.htgcrf import hetero_causal_forest


def test_htgcrf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    mono_mask = np.random.default_rng(42).normal(0, 1, 100)
    result = hetero_causal_forest(y, D, X, mono_mask)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_htgcrf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    mono_mask = np.random.default_rng(42).normal(0, 1, 100)
    result = hetero_causal_forest(y, D, X, mono_mask)
    assert isinstance(result, dict)
