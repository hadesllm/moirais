"""Tests for rng235.rangayyan_ch4_complex_log_of_product."""
import numpy as np
import pytest
from moirais.fn.rng235 import rangayyan_ch4_complex_log_of_product


def test_rng235_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    H = np.random.default_rng(42).normal(0, 1, 100)
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_complex_log_of_product(X, H, omega)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng235_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    H = np.random.default_rng(42).normal(0, 1, 100)
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_complex_log_of_product(X, H, omega)
    assert isinstance(result, dict)
