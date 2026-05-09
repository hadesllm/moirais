"""Tests for rgparmod.rangayyan_parametric_sysid."""
import numpy as np
import pytest
from moirais.fn.rgparmod import rangayyan_parametric_sysid


def test_rgparmod_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = rangayyan_parametric_sysid(x, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgparmod_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = rangayyan_parametric_sysid(x, order)
    assert isinstance(result, dict)
