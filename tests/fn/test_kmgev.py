"""Tests for kmgev.kamath_g_eval."""
import numpy as np
import pytest
from moirais.fn.kmgev import kamath_g_eval


def test_kmgev_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    rubric = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_g_eval(x, y, rubric, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmgev_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    rubric = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_g_eval(x, y, rubric, model)
    assert isinstance(result, dict)
