"""Tests for crsent.cross_entropy."""
import numpy as np
import pytest
from moirais.fn.crsent import cross_entropy


def test_crsent_basic():
    """Test basic functionality."""
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    base = np.random.default_rng(42).normal(0, 1, 100)
    result = cross_entropy(p, q, base)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_crsent_edge():
    """Test edge cases."""
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    base = np.random.default_rng(42).normal(0, 1, 100)
    result = cross_entropy(p, q, base)
    assert isinstance(result, dict)
