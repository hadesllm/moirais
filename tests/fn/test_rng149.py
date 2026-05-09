"""Tests for rng149.rangayyan_ch3_wiener_frequency_relation."""
import numpy as np
import pytest
from moirais.fn.rng149 import rangayyan_ch3_wiener_frequency_relation


def test_rng149_basic():
    """Test basic functionality."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    S_xx = np.random.default_rng(42).normal(0, 1, 100)
    S_xd = np.random.default_rng(42).normal(0, 1, 100)
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_wiener_frequency_relation(W, S_xx, S_xd, omega)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng149_edge():
    """Test edge cases."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    S_xx = np.random.default_rng(42).normal(0, 1, 100)
    S_xd = np.random.default_rng(42).normal(0, 1, 100)
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_wiener_frequency_relation(W, S_xx, S_xd, omega)
    assert isinstance(result, dict)
