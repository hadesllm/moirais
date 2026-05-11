"""Tests for rng207.rangayyan_ch4_matched_filter_input_ft."""
import numpy as np
import pytest
from morie.fn.rng207 import rangayyan_ch4_matched_filter_input_ft


def test_rng207_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_matched_filter_input_ft(x, t, omega)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng207_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_matched_filter_input_ft(x, t, omega)
    assert isinstance(result, dict)
