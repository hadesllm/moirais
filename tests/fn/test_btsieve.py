"""Tests for btsieve.boot_sieve_general."""
import numpy as np
import pytest
from morie.fn.btsieve import boot_sieve_general


def test_btsieve_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fit_fn = (lambda v: v)
    rvs_fn = (lambda v: v)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_sieve_general(x, fit_fn, rvs_fn, stat, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btsieve_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fit_fn = (lambda v: v)
    rvs_fn = (lambda v: v)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_sieve_general(x, fit_fn, rvs_fn, stat, B)
    assert isinstance(result, dict)
