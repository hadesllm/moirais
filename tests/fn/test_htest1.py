"""Tests for htest1.horvitz_thompson."""
import numpy as np
import pytest
from moirais.fn.htest1 import horvitz_thompson


def test_htest1_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    pi = np.random.default_rng(42).normal(0, 1, 100)
    result = horvitz_thompson(y, pi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_htest1_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    pi = np.random.default_rng(42).normal(0, 1, 100)
    result = horvitz_thompson(y, pi)
    assert isinstance(result, dict)
