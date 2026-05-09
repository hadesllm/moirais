"""Tests for rgztf.rangayyan_z_transform."""
import numpy as np
import pytest
from moirais.fn.rgztf import rangayyan_z_transform


def test_rgztf_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_z_transform(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgztf_edge():
    """Test edge cases."""
    result = rangayyan_z_transform(np.array([42.0]))
    assert result['n'] == 1
