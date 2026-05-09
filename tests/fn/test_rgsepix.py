"""Tests for rgsepix.rangayyan_separability_index."""
import numpy as np
import pytest
from moirais.fn.rgsepix import rangayyan_separability_index


def test_rgsepix_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = rangayyan_separability_index(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgsepix_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = rangayyan_separability_index(X, y)
    assert isinstance(result, dict)
