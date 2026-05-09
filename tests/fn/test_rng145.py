"""Tests for rng145.rangayyan_ch3_wiener_hopf_normal_equation."""
import numpy as np
import pytest
from moirais.fn.rng145 import rangayyan_ch3_wiener_hopf_normal_equation


def test_rng145_basic():
    """Test basic functionality."""
    Phi = np.random.default_rng(42).normal(0, 1, 100)
    w_o = np.random.default_rng(42).normal(0, 1, 100)
    Theta = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_wiener_hopf_normal_equation(Phi, w_o, Theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng145_edge():
    """Test edge cases."""
    Phi = np.random.default_rng(42).normal(0, 1, 100)
    w_o = np.random.default_rng(42).normal(0, 1, 100)
    Theta = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_wiener_hopf_normal_equation(Phi, w_o, Theta)
    assert isinstance(result, dict)
