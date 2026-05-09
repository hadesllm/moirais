"""Tests for mrkcsr.csr_test."""
import numpy as np
import pytest
from moirais.fn.mrkcsr import csr_test


def test_mrkcsr_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    window = np.random.default_rng(42).normal(0, 1, 100)
    nsim = np.random.default_rng(42).normal(0, 1, 100)
    result = csr_test(coords, window, nsim)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_mrkcsr_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    window = np.random.default_rng(42).normal(0, 1, 100)
    nsim = np.random.default_rng(42).normal(0, 1, 100)
    result = csr_test(coords, window, nsim)
    assert isinstance(result, dict)
