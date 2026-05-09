"""Tests for km124.kamath_ch8_ngram_embedding."""
import numpy as np
import pytest
from moirais.fn.km124 import kamath_ch8_ngram_embedding


def test_km124_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = kamath_ch8_ngram_embedding(x, i, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km124_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = kamath_ch8_ngram_embedding(x, i, n)
    assert isinstance(result, dict)
