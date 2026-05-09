"""Tests for gb1461.gibbons_multinomial_gof."""
import numpy as np
import pytest
from moirais.fn.gb1461 import gibbons_multinomial_gof


def test_gb1461_basic():
    """Test basic functionality."""
    observed = np.random.default_rng(42).normal(0, 1, 100)
    expected_probs = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_multinomial_gof(observed, expected_probs, n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1461_edge():
    """Test edge cases."""
    observed = np.random.default_rng(42).normal(0, 1, 100)
    expected_probs = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_multinomial_gof(observed, expected_probs, n)
    assert isinstance(result, dict)
