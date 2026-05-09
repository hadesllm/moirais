"""Tests for asorxx.assortativity."""
import numpy as np
import pytest
from moirais.fn.asorxx import assortativity


def test_asorxx_basic():
    """Test basic functionality."""
    G = np.eye(10)
    attribute = np.random.default_rng(42).normal(0, 1, 100)
    result = assortativity(G, attribute)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_asorxx_edge():
    """Test edge cases."""
    G = np.eye(10)
    attribute = np.random.default_rng(42).normal(0, 1, 100)
    result = assortativity(G, attribute)
    assert isinstance(result, dict)
