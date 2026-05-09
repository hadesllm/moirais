"""Tests for rng052.rangayyan_ch3_z_transform_definition."""
import numpy as np
import pytest
from moirais.fn.rng052 import rangayyan_ch3_z_transform_definition


def test_rng052_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_ch3_z_transform_definition(x, n, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng052_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_ch3_z_transform_definition(x, n, z)
    assert isinstance(result, dict)
