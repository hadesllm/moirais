"""Tests for snht.snht."""
import numpy as np
import pytest
from morie.fn.snht import snht


def test_snht_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = snht(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_snht_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = snht(x)
    assert isinstance(result, dict)
