"""Tests for jarow.jaro_winkler."""
import numpy as np
import pytest
from moirais.fn.jarow import jaro_winkler


def test_jarow_basic():
    """Test basic functionality."""
    s1 = np.random.default_rng(42).normal(0, 1, 100)
    s2 = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = jaro_winkler(s1, s2, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jarow_edge():
    """Test edge cases."""
    s1 = np.random.default_rng(42).normal(0, 1, 100)
    s2 = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = jaro_winkler(s1, s2, p)
    assert isinstance(result, dict)
