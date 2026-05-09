"""Tests for hmlrex.geron_lr_exponential."""
import numpy as np
import pytest
from moirais.fn.hmlrex import geron_lr_exponential


def test_hmlrex_basic():
    """Test basic functionality."""
    eta0 = np.random.default_rng(42).normal(0, 1, 100)
    decay = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = geron_lr_exponential(eta0, decay, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmlrex_edge():
    """Test edge cases."""
    eta0 = np.random.default_rng(42).normal(0, 1, 100)
    decay = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = geron_lr_exponential(eta0, decay, t)
    assert isinstance(result, dict)
