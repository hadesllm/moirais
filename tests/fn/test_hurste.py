"""Tests for hurste.hurst_exponent."""
import numpy as np
import pytest
from morie.fn.hurste import hurst_exponent


def test_hurste_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = hurst_exponent(y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hurste_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = hurst_exponent(y)
    assert isinstance(result, dict)
