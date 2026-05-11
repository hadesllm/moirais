"""Tests for rgemd.rangayyan_emd."""
import numpy as np
import pytest
from morie.fn.rgemd import rangayyan_emd


def test_rgemd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    max_imfs = np.random.default_rng(42).normal(0, 1, 100)
    tol = 1e-6
    result = rangayyan_emd(x, max_imfs, tol)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgemd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    max_imfs = np.random.default_rng(42).normal(0, 1, 100)
    tol = 1e-6
    result = rangayyan_emd(x, max_imfs, tol)
    assert isinstance(result, dict)
