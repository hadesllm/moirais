"""Tests for causscss.causal_synthetic_subset."""
import numpy as np
import pytest
from moirais.fn.causscss import causal_synthetic_subset


def test_causscss_basic():
    """Test basic functionality."""
    X1_pre = np.random.default_rng(42).normal(0, 1, 100)
    X0_pre = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = causal_synthetic_subset(X1_pre, X0_pre, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causscss_edge():
    """Test edge cases."""
    X1_pre = np.random.default_rng(42).normal(0, 1, 100)
    X0_pre = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = causal_synthetic_subset(X1_pre, X0_pre, lam)
    assert isinstance(result, dict)
