"""Tests for ebmZD.zonal_ebm."""
import numpy as np
import pytest
from moirais.fn.ebmZD import zonal_ebm


def test_ebmZD_basic():
    """Test basic functionality."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = zonal_ebm(S, alpha, A, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ebmZD_edge():
    """Test edge cases."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = zonal_ebm(S, alpha, A, B)
    assert isinstance(result, dict)
