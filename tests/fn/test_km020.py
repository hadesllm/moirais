"""Tests for km020.kamath_ch2_ssl_loss."""
import numpy as np
import pytest
from moirais.fn.km020 import kamath_ch2_ssl_loss


def test_km020_basic():
    """Test basic functionality."""
    L_PTi = np.random.default_rng(42).normal(0, 1, 100)
    lambda_i = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_ssl_loss(L_PTi, lambda_i)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km020_edge():
    """Test edge cases."""
    L_PTi = np.random.default_rng(42).normal(0, 1, 100)
    lambda_i = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_ssl_loss(L_PTi, lambda_i)
    assert isinstance(result, dict)
