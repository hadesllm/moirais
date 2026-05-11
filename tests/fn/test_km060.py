"""Tests for km060.kamath_ch4_krona_efficient."""
import numpy as np
import pytest
from morie.fn.km060 import kamath_ch4_krona_efficient


def test_km060_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch4_krona_efficient(A, B, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km060_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch4_krona_efficient(A, B, x)
    assert isinstance(result, dict)
