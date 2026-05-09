"""Tests for rng164.rangayyan_ch3_rls_normal_equation."""
import numpy as np
import pytest
from moirais.fn.rng164 import rangayyan_ch3_rls_normal_equation


def test_rng164_basic():
    """Test basic functionality."""
    Phi = np.random.default_rng(42).normal(0, 1, 100)
    w_tilde = np.random.default_rng(42).normal(0, 1, 100)
    Theta = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_rls_normal_equation(Phi, w_tilde, Theta, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng164_edge():
    """Test edge cases."""
    Phi = np.random.default_rng(42).normal(0, 1, 100)
    w_tilde = np.random.default_rng(42).normal(0, 1, 100)
    Theta = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_rls_normal_equation(Phi, w_tilde, Theta, n)
    assert isinstance(result, dict)
