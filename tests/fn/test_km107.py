"""Tests for km107.kamath_ch6_pii_likelihood."""
import numpy as np
import pytest
from moirais.fn.km107 import kamath_ch6_pii_likelihood


def test_km107_basic():
    """Test basic functionality."""
    a_m = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    x = np.random.default_rng(42).normal(0, 1, 100)
    L_q = np.random.default_rng(42).normal(0, 1, 100)
    L_r = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_pii_likelihood(a_m, A, x, L_q, L_r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km107_edge():
    """Test edge cases."""
    a_m = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    x = np.random.default_rng(42).normal(0, 1, 100)
    L_q = np.random.default_rng(42).normal(0, 1, 100)
    L_r = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_pii_likelihood(a_m, A, x, L_q, L_r)
    assert isinstance(result, dict)
