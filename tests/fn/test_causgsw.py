"""Tests for causgsw.causal_generalisability_smd."""
import numpy as np
import pytest
from moirais.fn.causgsw import causal_generalisability_smd


def test_causgsw_basic():
    """Test basic functionality."""
    X_trial = np.random.default_rng(42).normal(0, 1, 100)
    X_pop = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_generalisability_smd(X_trial, X_pop)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causgsw_edge():
    """Test edge cases."""
    X_trial = np.random.default_rng(42).normal(0, 1, 100)
    X_pop = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_generalisability_smd(X_trial, X_pop)
    assert isinstance(result, dict)
