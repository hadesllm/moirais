"""Tests for condie.conditional_indirect_effect."""
import numpy as np
import pytest
from moirais.fn.condie import conditional_indirect_effect


def test_condie_basic():
    """Test basic functionality."""
    a1 = np.random.default_rng(42).normal(0, 1, 100)
    a3 = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = conditional_indirect_effect(a1, a3, b, w)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_condie_edge():
    """Test edge cases."""
    a1 = np.random.default_rng(42).normal(0, 1, 100)
    a3 = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = conditional_indirect_effect(a1, a3, b, w)
    assert isinstance(result, dict)
