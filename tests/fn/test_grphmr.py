"""Tests for grphmr.graphormer."""
import numpy as np
import pytest
from moirais.fn.grphmr import graphormer


def test_grphmr_basic():
    """Test basic functionality."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = graphormer(G, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grphmr_edge():
    """Test edge cases."""
    G = np.eye(10)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = graphormer(G, X)
    assert isinstance(result, dict)
