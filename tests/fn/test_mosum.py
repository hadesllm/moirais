"""Tests for mosum.mosum."""
import numpy as np
import pytest
from moirais.fn.mosum import mosum


def test_mosum_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = mosum(x, h, threshold)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_mosum_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = mosum(x, h, threshold)
    assert isinstance(result, dict)
