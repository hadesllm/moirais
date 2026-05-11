"""Tests for rgbayng.rangayyan_bayes_gaussian."""
import numpy as np
import pytest
from morie.fn.rgbayng import rangayyan_bayes_gaussian


def test_rgbayng_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    mu_list = np.random.default_rng(42).normal(0, 1, 100)
    sigma_list = np.random.default_rng(42).normal(0, 1, 100)
    priors = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_bayes_gaussian(X, mu_list, sigma_list, priors)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgbayng_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    mu_list = np.random.default_rng(42).normal(0, 1, 100)
    sigma_list = np.random.default_rng(42).normal(0, 1, 100)
    priors = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_bayes_gaussian(X, mu_list, sigma_list, priors)
    assert isinstance(result, dict)
