"""Tests for km143.kamath_ch9_fom_loss."""
import numpy as np
import pytest
from moirais.fn.km143 import kamath_ch9_fom_loss


def test_km143_basic():
    """Test basic functionality."""
    r_i = np.random.default_rng(42).normal(0, 1, 100)
    t_i = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch9_fom_loss(r_i, t_i, R)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km143_edge():
    """Test edge cases."""
    r_i = np.random.default_rng(42).normal(0, 1, 100)
    t_i = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch9_fom_loss(r_i, t_i, R)
    assert isinstance(result, dict)
