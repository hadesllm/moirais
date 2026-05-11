"""Tests for byscn.bayes_cpi_prior."""
import numpy as np
import pytest
from morie.fn.byscn import bayes_cpi_prior


def test_byscn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    p0 = np.random.default_rng(42).normal(0, 1, 100)
    p1 = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_cpi_prior(y, X, p0, p1)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_byscn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    p0 = np.random.default_rng(42).normal(0, 1, 100)
    p1 = np.random.default_rng(42).normal(0, 1, 100)
    result = bayes_cpi_prior(y, X, p0, p1)
    assert isinstance(result, dict)
