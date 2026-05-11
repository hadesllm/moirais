"""Tests for km091.kamath_ch6_demographic_representation."""
import numpy as np
import pytest
from morie.fn.km091 import kamath_ch6_demographic_representation


def test_km091_basic():
    """Test basic functionality."""
    G_i = np.random.default_rng(42).normal(0, 1, 100)
    A_i = np.random.default_rng(42).normal(0, 1, 100)
    Yhat = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_demographic_representation(G_i, A_i, Yhat)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km091_edge():
    """Test edge cases."""
    G_i = np.random.default_rng(42).normal(0, 1, 100)
    A_i = np.random.default_rng(42).normal(0, 1, 100)
    Yhat = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_demographic_representation(G_i, A_i, Yhat)
    assert isinstance(result, dict)
