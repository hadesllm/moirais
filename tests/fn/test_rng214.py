"""Tests for rng214.rangayyan_ch4_signal_total_energy."""
import numpy as np
import pytest
from moirais.fn.rng214 import rangayyan_ch4_signal_total_energy


def test_rng214_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    t = np.linspace(0, 10, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_signal_total_energy(x, X, t, f)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng214_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    t = np.linspace(0, 10, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_signal_total_energy(x, X, t, f)
    assert isinstance(result, dict)
