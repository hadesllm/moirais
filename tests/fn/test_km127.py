"""Tests for km127.kamath_ch8_geval_score."""
import numpy as np
import pytest
from morie.fn.km127 import kamath_ch8_geval_score


def test_km127_basic():
    """Test basic functionality."""
    s_i = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = kamath_ch8_geval_score(s_i, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km127_edge():
    """Test edge cases."""
    s_i = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = kamath_ch8_geval_score(s_i, p)
    assert isinstance(result, dict)
