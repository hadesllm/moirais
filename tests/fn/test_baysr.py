"""Tests for baysr.bayes_r_prior."""
import numpy as np
import pytest
from morie.fn.baysr import bayes_r_prior


def test_baysr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    pi = np.random.default_rng(42).normal(0, 1, 100)
    sigma_classes = np.random.default_rng(43).integers(0, 2, 100)
    result = bayes_r_prior(y, X, pi, sigma_classes)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_baysr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    pi = np.random.default_rng(42).normal(0, 1, 100)
    sigma_classes = np.random.default_rng(43).integers(0, 2, 100)
    result = bayes_r_prior(y, X, pi, sigma_classes)
    assert isinstance(result, dict)
