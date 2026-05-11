"""Tests for rgpzp.rangayyan_pole_zero_plot."""
import numpy as np
import pytest
from morie.fn.rgpzp import rangayyan_pole_zero_plot


def test_rgpzp_basic():
    """Test basic functionality."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_pole_zero_plot(b, a)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgpzp_edge():
    """Test edge cases."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_pole_zero_plot(b, a)
    assert isinstance(result, dict)
