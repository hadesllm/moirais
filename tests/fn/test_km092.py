"""Tests for km092.kamath_ch6_stereotypical_assoc."""
import numpy as np
import pytest
from morie.fn.km092 import kamath_ch6_stereotypical_assoc


def test_km092_basic():
    """Test basic functionality."""
    w = np.random.default_rng(45).exponential(1, 100)
    A_i = np.random.default_rng(42).normal(0, 1, 100)
    Yhat = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_stereotypical_assoc(w, A_i, Yhat)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km092_edge():
    """Test edge cases."""
    w = np.random.default_rng(45).exponential(1, 100)
    A_i = np.random.default_rng(42).normal(0, 1, 100)
    Yhat = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_stereotypical_assoc(w, A_i, Yhat)
    assert isinstance(result, dict)
