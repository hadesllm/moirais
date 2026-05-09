"""Tests for chasym.check_asymptote_msm."""
import numpy as np
import pytest
from moirais.fn.chasym import check_asymptote_msm


def test_chasym_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = check_asymptote_msm(y, A, H, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_chasym_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = check_asymptote_msm(y, A, H, B)
    assert isinstance(result, dict)
