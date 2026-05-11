"""Tests for rng016.rangayyan_ch3_acf_continuous."""
import numpy as np
import pytest
from morie.fn.rng016 import rangayyan_ch3_acf_continuous


def test_rng016_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t1 = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = rangayyan_ch3_acf_continuous(x, t1, tau)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_rng016_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t1 = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = rangayyan_ch3_acf_continuous(x, t1, tau)
    assert isinstance(result, dict)
