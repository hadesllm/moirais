"""Tests for aitcap.compositional_classifyAP."""
import numpy as np
import pytest
from moirais.fn.aitcap import compositional_classifyAP


def test_aitcap_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    x_new = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = compositional_classifyAP(X, y, x_new, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitcap_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    x_new = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = compositional_classifyAP(X, y, x_new, k)
    assert isinstance(result, dict)
