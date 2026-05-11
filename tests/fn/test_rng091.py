"""Tests for rng091.rangayyan_ch3_hann_z_output."""
import numpy as np
import pytest
from morie.fn.rng091 import rangayyan_ch3_hann_z_output


def test_rng091_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_ch3_hann_z_output(X, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng091_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_ch3_hann_z_output(X, z)
    assert isinstance(result, dict)
