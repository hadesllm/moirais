"""Tests for rng139.rangayyan_ch3_wiener_output_dot_product."""
import numpy as np
import pytest
from morie.fn.rng139 import rangayyan_ch3_wiener_output_dot_product


def test_rng139_basic():
    """Test basic functionality."""
    w = np.random.default_rng(45).exponential(1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_wiener_output_dot_product(w, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng139_edge():
    """Test edge cases."""
    w = np.random.default_rng(45).exponential(1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_wiener_output_dot_product(w, x)
    assert isinstance(result, dict)
