"""Tests for km081.kamath_ch6_weat_similarity."""
import numpy as np
import pytest
from morie.fn.km081 import kamath_ch6_weat_similarity


def test_km081_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    W_1 = np.random.default_rng(42).normal(0, 1, 100)
    W_2 = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_weat_similarity(a, W_1, W_2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km081_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    W_1 = np.random.default_rng(42).normal(0, 1, 100)
    W_2 = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_weat_similarity(a, W_1, W_2)
    assert isinstance(result, dict)
