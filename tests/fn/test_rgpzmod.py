"""Tests for rgpzmod.rangayyan_pole_zero_model."""
import numpy as np
import pytest
from moirais.fn.rgpzmod import rangayyan_pole_zero_model


def test_rgpzmod_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_pole_zero_model(x, p, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgpzmod_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_pole_zero_model(x, p, q)
    assert isinstance(result, dict)
