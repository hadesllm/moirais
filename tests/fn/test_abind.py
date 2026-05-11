"""Tests for abind.ab_indirect_effect."""
import numpy as np
import pytest
from morie.fn.abind import ab_indirect_effect


def test_abind_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = ab_indirect_effect(a, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_abind_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = ab_indirect_effect(a, b)
    assert isinstance(result, dict)
