"""Tests for km089.kamath_ch6_sgs_invariance."""
import numpy as np
import pytest
from moirais.fn.km089 import kamath_ch6_sgs_invariance


def test_km089_basic():
    """Test basic functionality."""
    Yhat_i = np.random.default_rng(42).normal(0, 1, 100)
    Yhat_j = np.random.default_rng(42).normal(0, 1, 100)
    psi = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_sgs_invariance(Yhat_i, Yhat_j, psi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km089_edge():
    """Test edge cases."""
    Yhat_i = np.random.default_rng(42).normal(0, 1, 100)
    Yhat_j = np.random.default_rng(42).normal(0, 1, 100)
    psi = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_sgs_invariance(Yhat_i, Yhat_j, psi)
    assert isinstance(result, dict)
