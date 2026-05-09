"""Tests for magsd.ma_glass_delta."""
import numpy as np
import pytest
from moirais.fn.magsd import ma_glass_delta


def test_magsd_basic():
    """Test basic functionality."""
    m1 = np.random.default_rng(42).normal(0, 1, 100)
    m2 = np.random.default_rng(42).normal(0, 1, 100)
    s_ctrl = np.random.default_rng(42).normal(0, 1, 100)
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_glass_delta(m1, m2, s_ctrl, n1, n2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_magsd_edge():
    """Test edge cases."""
    m1 = np.random.default_rng(42).normal(0, 1, 100)
    m2 = np.random.default_rng(42).normal(0, 1, 100)
    s_ctrl = np.random.default_rng(42).normal(0, 1, 100)
    n1 = np.random.default_rng(42).normal(0, 1, 100)
    n2 = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_glass_delta(m1, m2, s_ctrl, n1, n2)
    assert isinstance(result, dict)
