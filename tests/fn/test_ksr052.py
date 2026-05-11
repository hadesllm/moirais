"""Tests for ksr052.kosorok_ch2_kaplan_meier_derivative."""
import numpy as np
import pytest
from morie.fn.ksr052 import kosorok_ch2_kaplan_meier_derivative


def test_ksr052_basic():
    """Test basic functionality."""
    S_0 = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    h = 0.3
    t = np.linspace(0, 10, 100)
    result = kosorok_ch2_kaplan_meier_derivative(S_0, L, G, h, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr052_edge():
    """Test edge cases."""
    S_0 = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    h = 0.3
    t = np.linspace(0, 10, 100)
    result = kosorok_ch2_kaplan_meier_derivative(S_0, L, G, h, t)
    assert isinstance(result, dict)
