"""Tests for csrkstst.kstest_csr."""
import numpy as np
import pytest
from morie.fn.csrkstst import kstest_csr


def test_csrkstst_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = kstest_csr(coords, window)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_csrkstst_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = kstest_csr(coords, window)
    assert isinstance(result, dict)
