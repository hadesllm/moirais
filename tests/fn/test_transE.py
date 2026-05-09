"""Tests for transE.transe."""
import numpy as np
import pytest
from moirais.fn.transE import transe


def test_transE_basic():
    """Test basic functionality."""
    triples = np.random.default_rng(42).normal(0, 1, 100)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = transe(triples, dim)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_transE_edge():
    """Test edge cases."""
    triples = np.random.default_rng(42).normal(0, 1, 100)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = transe(triples, dim)
    assert isinstance(result, dict)
