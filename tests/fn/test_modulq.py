"""Tests for modulq.modularity_q."""
import numpy as np
import pytest
from moirais.fn.modulq import modularity_q


def test_modulq_basic():
    """Test basic functionality."""
    G = np.eye(10)
    communities = np.random.default_rng(42).normal(0, 1, 100)
    result = modularity_q(G, communities)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_modulq_edge():
    """Test edge cases."""
    G = np.eye(10)
    communities = np.random.default_rng(42).normal(0, 1, 100)
    result = modularity_q(G, communities)
    assert isinstance(result, dict)
