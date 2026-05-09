"""Tests for csfgrf.causal_survival_forest."""
import numpy as np
import pytest
from moirais.fn.csfgrf import causal_survival_forest


def test_csfgrf_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = causal_survival_forest(time, event, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_csfgrf_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = causal_survival_forest(time, event, D, X)
    assert isinstance(result, dict)
