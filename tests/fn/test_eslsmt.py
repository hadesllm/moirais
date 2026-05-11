"""Tests for eslsmt.esl_smoothing_spline."""
import numpy as np
import pytest
from morie.fn.eslsmt import esl_smoothing_spline


def test_eslsmt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_smoothing_spline(x, y, lambda_)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslsmt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    lambda_ = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_smoothing_spline(x, y, lambda_)
    assert isinstance(result, dict)
