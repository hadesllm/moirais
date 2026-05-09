"""Tests for km062.kamath_ch4_krona_tuned_weights."""
import numpy as np
import pytest
from moirais.fn.km062 import kamath_ch4_krona_tuned_weights


def test_km062_basic():
    """Test basic functionality."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    A_k = np.random.default_rng(42).normal(0, 1, 100)
    B_k = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    result = kamath_ch4_krona_tuned_weights(W, A_k, B_k, s)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km062_edge():
    """Test edge cases."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    A_k = np.random.default_rng(42).normal(0, 1, 100)
    B_k = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    result = kamath_ch4_krona_tuned_weights(W, A_k, B_k, s)
    assert isinstance(result, dict)
