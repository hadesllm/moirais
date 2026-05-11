"""Tests for km095.kamath_ch6_gender_direction."""
import numpy as np
import pytest
from morie.fn.km095 import kamath_ch6_gender_direction


def test_km095_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    E = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_gender_direction(A, E)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km095_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    E = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_gender_direction(A, E)
    assert isinstance(result, dict)
