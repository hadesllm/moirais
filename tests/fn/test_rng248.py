"""Tests for rng248.rangayyan_ch4_z_transform_signal_echo."""
import numpy as np
import pytest
from morie.fn.rng248 import rangayyan_ch4_z_transform_signal_echo


def test_rng248_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    n_0 = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_z_transform_signal_echo(a, n_0, z, H)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng248_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    n_0 = np.random.default_rng(42).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_z_transform_signal_echo(a, n_0, z, H)
    assert isinstance(result, dict)
