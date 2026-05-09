"""Tests for frmlD.formal_derivative."""
import numpy as np
import pytest
from moirais.fn.frmlD import formal_derivative


def test_frmlD_basic():
    """Test basic functionality."""
    poly = np.random.default_rng(42).normal(0, 1, 100)
    result = formal_derivative(poly)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_frmlD_edge():
    """Test edge cases."""
    poly = np.random.default_rng(42).normal(0, 1, 100)
    result = formal_derivative(poly)
    assert isinstance(result, dict)
