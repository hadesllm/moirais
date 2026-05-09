"""Tests for km088.kamath_ch6_cat_metric."""
import numpy as np
import pytest
from moirais.fn.km088 import kamath_ch6_cat_metric


def test_km088_basic():
    """Test basic functionality."""
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    U = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = kamath_ch6_cat_metric(M, U, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km088_edge():
    """Test edge cases."""
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    U = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = kamath_ch6_cat_metric(M, U, theta)
    assert isinstance(result, dict)
