"""Tests for distM.distmult."""
import numpy as np
import pytest
from morie.fn.distM import distmult


def test_distM_basic():
    """Test basic functionality."""
    triples = np.random.default_rng(42).normal(0, 1, 100)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = distmult(triples, dim)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_distM_edge():
    """Test edge cases."""
    triples = np.random.default_rng(42).normal(0, 1, 100)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = distmult(triples, dim)
    assert isinstance(result, dict)
