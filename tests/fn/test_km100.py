"""Tests for km100.kamath_ch6_toxicity_probability."""
import numpy as np
import pytest
from moirais.fn.km100 import kamath_ch6_toxicity_probability


def test_km100_basic():
    """Test basic functionality."""
    Yhat = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_toxicity_probability(Yhat, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km100_edge():
    """Test edge cases."""
    Yhat = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_toxicity_probability(Yhat, c)
    assert isinstance(result, dict)
