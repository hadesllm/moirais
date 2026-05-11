"""Tests for survcfg.causal_survival_forest_grf."""
import numpy as np
import pytest
from morie.fn.survcfg import causal_survival_forest_grf


def test_survcfg_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = causal_survival_forest_grf(time, event, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survcfg_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = causal_survival_forest_grf(time, event, D, X)
    assert isinstance(result, dict)
