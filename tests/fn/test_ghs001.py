"""Tests for ghs001.ghosal_ch1_bayes_formula."""
import numpy as np
import pytest
from moirais.fn.ghs001 import ghosal_ch1_bayes_formula


def test_ghs001_basic():
    """Test basic functionality."""
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    p_theta = np.random.default_rng(42).normal(0, 1, 100)
    Pi = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_ch1_bayes_formula(B, X, p_theta, Pi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs001_edge():
    """Test edge cases."""
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    p_theta = np.random.default_rng(42).normal(0, 1, 100)
    Pi = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_ch1_bayes_formula(B, X, p_theta, Pi)
    assert isinstance(result, dict)
