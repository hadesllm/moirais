"""Tests for rng161.rangayyan_ch3_lms_variable_step."""
import numpy as np
import pytest
from morie.fn.rng161 import rangayyan_ch3_lms_variable_step


def test_rng161_basic():
    """Test basic functionality."""
    w = np.random.default_rng(45).exponential(1, 100)
    mu = 0.0
    e = np.random.default_rng(44).normal(0, 1, 100)
    r = 10
    n = 100
    result = rangayyan_ch3_lms_variable_step(w, mu, e, r, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng161_edge():
    """Test edge cases."""
    w = np.random.default_rng(45).exponential(1, 100)
    mu = 0.0
    e = np.random.default_rng(44).normal(0, 1, 100)
    r = 10
    n = 100
    result = rangayyan_ch3_lms_variable_step(w, mu, e, r, n)
    assert isinstance(result, dict)
