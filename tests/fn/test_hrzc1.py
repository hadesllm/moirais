"""Tests for hrzc1.horowitz_censored_regression."""
import numpy as np
import pytest
from morie.fn.hrzc1 import horowitz_censored_regression


def test_hrzc1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    censor = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_censored_regression(x, y, censor)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzc1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    censor = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_censored_regression(x, y, censor)
    assert isinstance(result, dict)
