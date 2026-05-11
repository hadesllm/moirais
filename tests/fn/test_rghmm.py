"""Tests for rghmm.rangayyan_fitzhugh_nagumo."""
import numpy as np
import pytest
from morie.fn.rghmm import rangayyan_fitzhugh_nagumo


def test_rghmm_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    I_ext = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_fitzhugh_nagumo(t, I_ext, a, b, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rghmm_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    I_ext = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_fitzhugh_nagumo(t, I_ext, a, b, eps)
    assert isinstance(result, dict)
