"""Tests for rgerrbd.rangayyan_bayes_error_bound."""
import numpy as np
import pytest
from morie.fn.rgerrbd import rangayyan_bayes_error_bound


def test_rgerrbd_basic():
    """Test basic functionality."""
    mu1 = np.random.default_rng(42).normal(0, 1, 100)
    sigma1 = np.random.default_rng(42).normal(0, 1, 100)
    p1 = np.random.default_rng(42).normal(0, 1, 100)
    mu2 = np.random.default_rng(42).normal(0, 1, 100)
    sigma2 = np.random.default_rng(42).normal(0, 1, 100)
    p2 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_bayes_error_bound(mu1, sigma1, p1, mu2, sigma2, p2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgerrbd_edge():
    """Test edge cases."""
    mu1 = np.random.default_rng(42).normal(0, 1, 100)
    sigma1 = np.random.default_rng(42).normal(0, 1, 100)
    p1 = np.random.default_rng(42).normal(0, 1, 100)
    mu2 = np.random.default_rng(42).normal(0, 1, 100)
    sigma2 = np.random.default_rng(42).normal(0, 1, 100)
    p2 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_bayes_error_bound(mu1, sigma1, p1, mu2, sigma2, p2)
    assert isinstance(result, dict)
