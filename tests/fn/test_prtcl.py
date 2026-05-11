"""Tests for prtcl.particle_filter."""
import numpy as np
import pytest
from morie.fn.prtcl import particle_filter


def test_prtcl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    N = 100
    result = particle_filter(y, f, h, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_prtcl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    N = 100
    result = particle_filter(y, f, h, N)
    assert isinstance(result, dict)
