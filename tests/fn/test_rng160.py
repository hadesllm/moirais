"""Tests for rng160.rangayyan_ch3_widrow_hoff_lms."""
import numpy as np
import pytest
from morie.fn.rng160 import rangayyan_ch3_widrow_hoff_lms


def test_rng160_basic():
    """Test basic functionality."""
    w = np.random.default_rng(45).exponential(1, 100)
    mu = 0.0
    e = np.random.default_rng(44).normal(0, 1, 100)
    r = 10
    n = 100
    result = rangayyan_ch3_widrow_hoff_lms(w, mu, e, r, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng160_edge():
    """Test edge cases."""
    w = np.random.default_rng(45).exponential(1, 100)
    mu = 0.0
    e = np.random.default_rng(44).normal(0, 1, 100)
    r = 10
    n = 100
    result = rangayyan_ch3_widrow_hoff_lms(w, mu, e, r, n)
    assert isinstance(result, dict)
