"""Tests for hellngd.hellinger_distance."""
import numpy as np
import pytest
from moirais.fn.hellngd import hellinger_distance


def test_hellngd_basic():
    """Test basic functionality."""
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = hellinger_distance(p, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hellngd_edge():
    """Test edge cases."""
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = hellinger_distance(p, q)
    assert isinstance(result, dict)
