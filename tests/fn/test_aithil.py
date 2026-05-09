"""Tests for aithil.compositional_hill."""
import numpy as np
import pytest
from moirais.fn.aithil import compositional_hill


def test_aithil_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = compositional_hill(x, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aithil_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = compositional_hill(x, q)
    assert isinstance(result, dict)
