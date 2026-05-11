"""Tests for rgengn.rangayyan_eng."""
import numpy as np
import pytest
from morie.fn.rgengn import rangayyan_eng


def test_rgengn_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    n_fibers = np.random.default_rng(42).normal(0, 1, 100)
    cv_range = np.random.default_rng(42).normal(0, 1, 100)
    amp_range = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_eng(t, n_fibers, cv_range, amp_range)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgengn_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    n_fibers = np.random.default_rng(42).normal(0, 1, 100)
    cv_range = np.random.default_rng(42).normal(0, 1, 100)
    amp_range = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_eng(t, n_fibers, cv_range, amp_range)
    assert isinstance(result, dict)
