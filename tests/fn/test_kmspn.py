"""Tests for kmspn.kamath_t5_span_corruption."""
import numpy as np
import pytest
from morie.fn.kmspn import kamath_t5_span_corruption


def test_kmspn_basic():
    """Test basic functionality."""
    tokens = np.random.default_rng(42).normal(0, 1, 100)
    mean_span_len = np.random.default_rng(42).normal(0, 1, 100)
    corruption_rate = 0.1
    result = kamath_t5_span_corruption(tokens, mean_span_len, corruption_rate)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmspn_edge():
    """Test edge cases."""
    tokens = np.random.default_rng(42).normal(0, 1, 100)
    mean_span_len = np.random.default_rng(42).normal(0, 1, 100)
    corruption_rate = 0.1
    result = kamath_t5_span_corruption(tokens, mean_span_len, corruption_rate)
    assert isinstance(result, dict)
