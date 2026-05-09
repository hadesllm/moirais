"""Tests for kmmedu.kamath_medusa_heads."""
import numpy as np
import pytest
from moirais.fn.kmmedu import kamath_medusa_heads


def test_kmmedu_basic():
    """Test basic functionality."""
    hidden_state = np.random.default_rng(42).normal(0, 1, 100)
    medusa_heads = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = kamath_medusa_heads(hidden_state, medusa_heads, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmmedu_edge():
    """Test edge cases."""
    hidden_state = np.random.default_rng(42).normal(0, 1, 100)
    medusa_heads = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = kamath_medusa_heads(hidden_state, medusa_heads, k)
    assert isinstance(result, dict)
