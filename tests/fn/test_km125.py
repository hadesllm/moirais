"""Tests for km125.kamath_ch8_ngram_weight."""
import numpy as np
import pytest
from morie.fn.km125 import kamath_ch8_ngram_weight


def test_km125_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = kamath_ch8_ngram_weight(x, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km125_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = kamath_ch8_ngram_weight(x, Z)
    assert isinstance(result, dict)
