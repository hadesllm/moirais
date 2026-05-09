"""Tests for grlrex.geron_lr_exponential_schedule."""
import numpy as np
import pytest
from moirais.fn.grlrex import geron_lr_exponential_schedule


def test_grlrex_basic():
    """Test basic functionality."""
    eta0 = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    t = np.linspace(0, 10, 100)
    result = geron_lr_exponential_schedule(eta0, gamma, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grlrex_edge():
    """Test edge cases."""
    eta0 = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    t = np.linspace(0, 10, 100)
    result = geron_lr_exponential_schedule(eta0, gamma, t)
    assert isinstance(result, dict)
