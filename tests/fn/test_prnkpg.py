"""Tests for prnkpg.pagerank."""
import numpy as np
import pytest
from moirais.fn.prnkpg import pagerank


def test_prnkpg_basic():
    """Test basic functionality."""
    G = np.eye(10)
    damping = np.random.default_rng(42).normal(0, 1, 100)
    result = pagerank(G, damping)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_prnkpg_edge():
    """Test edge cases."""
    G = np.eye(10)
    damping = np.random.default_rng(42).normal(0, 1, 100)
    result = pagerank(G, damping)
    assert isinstance(result, dict)
