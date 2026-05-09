"""Tests for luvR.louvain."""
import numpy as np
import pytest
from moirais.fn.luvR import louvain


def test_luvR_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    resolution = np.random.default_rng(42).normal(0, 1, 100)
    result = louvain(A, resolution)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_luvR_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    resolution = np.random.default_rng(42).normal(0, 1, 100)
    result = louvain(A, resolution)
    assert isinstance(result, dict)
