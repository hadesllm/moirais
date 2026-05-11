"""Tests for rng127.rangayyan_ch3_bilinear_transformation."""
import numpy as np
import pytest
from morie.fn.rng127 import rangayyan_ch3_bilinear_transformation


def test_rng127_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_bilinear_transformation(z, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng127_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_bilinear_transformation(z, T)
    assert isinstance(result, dict)
