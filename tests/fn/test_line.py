"""Tests for line.line."""
import numpy as np
import pytest
from morie.fn.line import line


def test_line_basic():
    """Test basic functionality."""
    G = np.eye(10)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = line(G, dim)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_line_edge():
    """Test edge cases."""
    G = np.eye(10)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = line(G, dim)
    assert isinstance(result, dict)
