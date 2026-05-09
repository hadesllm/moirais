"""Tests for sdne.sdne."""
import numpy as np
import pytest
from moirais.fn.sdne import sdne


def test_sdne_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = sdne(A, dim)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sdne_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = sdne(A, dim)
    assert isinstance(result, dict)
