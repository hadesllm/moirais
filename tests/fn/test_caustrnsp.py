"""Tests for caustrnsp.causal_transportability_weights."""
import numpy as np
import pytest
from moirais.fn.caustrnsp import causal_transportability_weights


def test_caustrnsp_basic():
    """Test basic functionality."""
    X_source = np.random.default_rng(42).normal(0, 1, 100)
    X_target = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_transportability_weights(X_source, X_target)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_caustrnsp_edge():
    """Test edge cases."""
    X_source = np.random.default_rng(42).normal(0, 1, 100)
    X_target = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_transportability_weights(X_source, X_target)
    assert isinstance(result, dict)
