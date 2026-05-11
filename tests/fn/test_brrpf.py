"""Tests for brrpf.brr_prior_posterior."""
import numpy as np
import pytest
from morie.fn.brrpf import brr_prior_posterior


def test_brrpf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    a_b = np.random.default_rng(42).normal(0, 1, 100)
    b_b = np.random.default_rng(42).normal(0, 1, 100)
    a_e = np.random.default_rng(42).normal(0, 1, 100)
    b_e = np.random.default_rng(42).normal(0, 1, 100)
    result = brr_prior_posterior(y, X, a_b, b_b, a_e, b_e)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_brrpf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    a_b = np.random.default_rng(42).normal(0, 1, 100)
    b_b = np.random.default_rng(42).normal(0, 1, 100)
    a_e = np.random.default_rng(42).normal(0, 1, 100)
    b_e = np.random.default_rng(42).normal(0, 1, 100)
    result = brr_prior_posterior(y, X, a_b, b_b, a_e, b_e)
    assert isinstance(result, dict)
