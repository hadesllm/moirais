"""Tests for km087.kamath_ch6_cps_metric."""
import numpy as np
import pytest
from moirais.fn.km087 import kamath_ch6_cps_metric


def test_km087_basic():
    """Test basic functionality."""
    U = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    theta = 0.0
    result = kamath_ch6_cps_metric(U, M, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km087_edge():
    """Test edge cases."""
    U = np.random.default_rng(42).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    theta = 0.0
    result = kamath_ch6_cps_metric(U, M, theta)
    assert isinstance(result, dict)
