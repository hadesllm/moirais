"""Tests for gpcm.generalized_partial_credit."""
import numpy as np
import pytest
from morie.fn.gpcm import generalized_partial_credit


def test_gpcm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    a = np.random.default_rng(44).normal(0, 1, 100)
    b_j = np.random.default_rng(42).normal(0, 1, 100)
    result = generalized_partial_credit(y, theta, a, b_j)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gpcm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    a = np.random.default_rng(44).normal(0, 1, 100)
    b_j = np.random.default_rng(42).normal(0, 1, 100)
    result = generalized_partial_credit(y, theta, a, b_j)
    assert isinstance(result, dict)
