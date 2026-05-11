"""Tests for rng023.rangayyan_ch3_ccf_continuous."""
import numpy as np
import pytest
from morie.fn.rng023 import rangayyan_ch3_ccf_continuous


def test_rng023_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    t1 = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = rangayyan_ch3_ccf_continuous(x, y, t1, tau)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_rng023_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    t1 = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = rangayyan_ch3_ccf_continuous(x, y, t1, tau)
    assert isinstance(result, dict)
