"""Tests for btsubrho.boot_subsample_rate."""
import numpy as np
import pytest
from moirais.fn.btsubrho import boot_subsample_rate


def test_btsubrho_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    m_grid = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_subsample_rate(x, stat, m_grid, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btsubrho_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    m_grid = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_subsample_rate(x, stat, m_grid, B)
    assert isinstance(result, dict)
