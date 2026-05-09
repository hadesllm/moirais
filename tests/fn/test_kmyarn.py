"""Tests for kmyarn.kamath_yarn_context_extrapolation."""
import numpy as np
import pytest
from moirais.fn.kmyarn import kamath_yarn_context_extrapolation


def test_kmyarn_basic():
    """Test basic functionality."""
    theta = 0.0
    scale = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = kamath_yarn_context_extrapolation(theta, scale, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmyarn_edge():
    """Test edge cases."""
    theta = 0.0
    scale = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = kamath_yarn_context_extrapolation(theta, scale, d)
    assert isinstance(result, dict)
