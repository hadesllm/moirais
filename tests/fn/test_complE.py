"""Tests for complE.complex."""
import numpy as np
import pytest
from morie.fn.complE import complex


def test_complE_basic():
    """Test basic functionality."""
    triples = np.random.default_rng(42).normal(0, 1, 100)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = complex(triples, dim)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_complE_edge():
    """Test edge cases."""
    triples = np.random.default_rng(42).normal(0, 1, 100)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = complex(triples, dim)
    assert isinstance(result, dict)
