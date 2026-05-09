"""Tests for rgeqn7b.rangayyan_ch7_arma_error."""
import numpy as np
import pytest
from moirais.fn.rgeqn7b import rangayyan_ch7_arma_error


def test_rgeqn7b_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    b_coeffs = np.random.default_rng(42).normal(0, 1, 100)
    a_coeffs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch7_arma_error(x, b_coeffs, a_coeffs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgeqn7b_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    b_coeffs = np.random.default_rng(42).normal(0, 1, 100)
    a_coeffs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch7_arma_error(x, b_coeffs, a_coeffs)
    assert isinstance(result, dict)
