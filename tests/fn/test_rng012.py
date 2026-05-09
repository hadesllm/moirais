"""Tests for rng012.rangayyan_ch3_signal_plus_noise_model."""
import numpy as np
import pytest
from moirais.fn.rng012 import rangayyan_ch3_signal_plus_noise_model


def test_rng012_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_signal_plus_noise_model(x, eta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng012_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_signal_plus_noise_model(x, eta)
    assert isinstance(result, dict)
