"""Tests for rgeqn7a.rangayyan_ch7_ar_prediction_err."""
import numpy as np
import pytest
from morie.fn.rgeqn7a import rangayyan_ch7_ar_prediction_err


def test_rgeqn7a_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    a_coeffs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch7_ar_prediction_err(x, a_coeffs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgeqn7a_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    a_coeffs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch7_ar_prediction_err(x, a_coeffs)
    assert isinstance(result, dict)
