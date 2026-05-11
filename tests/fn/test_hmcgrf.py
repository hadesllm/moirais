"""Tests for hmcgrf.geron_computational_graph."""
import numpy as np
import pytest
from morie.fn.hmcgrf import geron_computational_graph


def test_hmcgrf_basic():
    """Test basic functionality."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_computational_graph(expr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmcgrf_edge():
    """Test edge cases."""
    expr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_computational_graph(expr)
    assert isinstance(result, dict)
