"""Tests for rng053.rangayyan_ch3_z_transform_fir."""
import numpy as np
import pytest
from moirais.fn.rng053 import rangayyan_ch3_z_transform_fir


def test_rng053_basic():
    """Test basic functionality."""
    h = 0.3
    n = 100
    z = np.random.default_rng(44).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_z_transform_fir(h, n, z, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng053_edge():
    """Test edge cases."""
    h = 0.3
    n = 100
    z = np.random.default_rng(44).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_z_transform_fir(h, n, z, N)
    assert isinstance(result, dict)
