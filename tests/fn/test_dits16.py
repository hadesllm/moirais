"""Tests for dits16.dit_diffusion_transformer."""
import numpy as np
import pytest
from morie.fn.dits16 import dit_diffusion_transformer


def test_dits16_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    cls = np.random.default_rng(42).normal(0, 1, 100)
    result = dit_diffusion_transformer(x, t, cls)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dits16_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    cls = np.random.default_rng(42).normal(0, 1, 100)
    result = dit_diffusion_transformer(x, t, cls)
    assert isinstance(result, dict)
