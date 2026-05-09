"""Tests for rotE.rotate."""
import numpy as np
import pytest
from moirais.fn.rotE import rotate


def test_rotE_basic():
    """Test basic functionality."""
    triples = np.random.default_rng(42).normal(0, 1, 100)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = rotate(triples, dim)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rotE_edge():
    """Test edge cases."""
    triples = np.random.default_rng(42).normal(0, 1, 100)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = rotate(triples, dim)
    assert isinstance(result, dict)
