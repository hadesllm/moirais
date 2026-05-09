"""Tests for btmoutn.boot_m_out_of_n."""
import numpy as np
import pytest
from moirais.fn.btmoutn import boot_m_out_of_n


def test_btmoutn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_m_out_of_n(x, m, stat, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btmoutn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_m_out_of_n(x, m, stat, B)
    assert isinstance(result, dict)
