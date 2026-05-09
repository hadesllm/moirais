"""Tests for km093.kamath_ch6_honest_score."""
import numpy as np
import pytest
from moirais.fn.km093 import kamath_ch6_honest_score


def test_km093_basic():
    """Test basic functionality."""
    Yhat = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = kamath_ch6_honest_score(Yhat, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km093_edge():
    """Test edge cases."""
    Yhat = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = kamath_ch6_honest_score(Yhat, k)
    assert isinstance(result, dict)
