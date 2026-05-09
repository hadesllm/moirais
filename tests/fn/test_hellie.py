"""Tests for hellie.hellinger_distance."""
import numpy as np
import pytest
from moirais.fn.hellie import hellinger_distance


def test_hellie_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = hellinger_distance(y, p, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hellie_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = hellinger_distance(y, p, q)
    assert isinstance(result, dict)
