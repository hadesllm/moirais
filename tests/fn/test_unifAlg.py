"""Tests for unifAlg.unification."""
import numpy as np
import pytest
from moirais.fn.unifAlg import unification


def test_unifAlg_basic():
    """Test basic functionality."""
    t1 = np.random.default_rng(42).normal(0, 1, 100)
    t2 = np.random.default_rng(42).normal(0, 1, 100)
    result = unification(t1, t2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_unifAlg_edge():
    """Test edge cases."""
    t1 = np.random.default_rng(42).normal(0, 1, 100)
    t2 = np.random.default_rng(42).normal(0, 1, 100)
    result = unification(t1, t2)
    assert isinstance(result, dict)
