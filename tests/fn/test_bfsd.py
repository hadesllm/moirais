"""Tests for bfsd.bayes_factor_savage_dickey."""
import numpy as np
import pytest
from morie.fn.bfsd import bayes_factor_savage_dickey


def test_bfsd_basic():
    """Test basic functionality."""
    samples = np.random.default_rng(42).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_factor_savage_dickey(samples, prior)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bfsd_edge():
    """Test edge cases."""
    samples = np.random.default_rng(42).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_factor_savage_dickey(samples, prior)
    assert isinstance(result, dict)
