"""Tests for prsmtd.propensity_score_method."""
import numpy as np
import pytest
from morie.fn.prsmtd import propensity_score_method


def test_prsmtd_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = propensity_score_method(A, H, time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_prsmtd_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    result = propensity_score_method(A, H, time)
    assert isinstance(result, dict)
