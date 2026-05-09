"""Tests for otreg.ot_regularised_dual."""
import numpy as np
import pytest
from moirais.fn.otreg import ot_regularised_dual


def test_otreg_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_regularised_dual(a, b, C, epsilon, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otreg_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_regularised_dual(a, b, C, epsilon, max_iter)
    assert isinstance(result, dict)
