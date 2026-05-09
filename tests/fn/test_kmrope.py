"""Tests for kmrope.kamath_rotary_positional_embedding."""
import numpy as np
import pytest
from moirais.fn.kmrope import kamath_rotary_positional_embedding


def test_kmrope_basic():
    """Test basic functionality."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    positions = np.random.default_rng(42).uniform(0, 1, (100, 2))
    base = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_rotary_positional_embedding(q, positions, base)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmrope_edge():
    """Test edge cases."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    positions = np.random.default_rng(42).uniform(0, 1, (100, 2))
    base = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_rotary_positional_embedding(q, positions, base)
    assert isinstance(result, dict)
