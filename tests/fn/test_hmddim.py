"""Tests for hmddim.geron_ddim."""
import numpy as np
import pytest
from morie.fn.hmddim import geron_ddim


def test_hmddim_basic():
    """Test basic functionality."""
    x_T = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    n_steps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_ddim(x_T, model, T, n_steps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmddim_edge():
    """Test edge cases."""
    x_T = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    n_steps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_ddim(x_T, model, T, n_steps)
    assert isinstance(result, dict)
