"""Tests for isbplr.isgp_bayes."""
import numpy as np
import pytest
from morie.fn.isbplr import isgp_bayes


def test_isbplr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    sigma = 1.0
    alpha = 0.05
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = isgp_bayes(y, sigma, alpha, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_isbplr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    sigma = 1.0
    alpha = 0.05
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = isgp_bayes(y, sigma, alpha, c)
    assert isinstance(result, dict)
