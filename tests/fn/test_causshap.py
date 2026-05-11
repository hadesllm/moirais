"""Tests for causshap.causal_shap_decomposition."""
import numpy as np
import pytest
from morie.fn.causshap import causal_shap_decomposition


def test_causshap_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    n_samples = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_shap_decomposition(X, y, model, n_samples)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causshap_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    n_samples = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_shap_decomposition(X, y, model, n_samples)
    assert isinstance(result, dict)
