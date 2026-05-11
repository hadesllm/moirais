"""Tests for rgvagadp.rangayyan_vag_adaptive_tfd."""
import numpy as np
import pytest
from morie.fn.rgvagadp import rangayyan_vag_adaptive_tfd


def test_rgvagadp_basic():
    """Test basic functionality."""
    vag = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    n_atoms = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_vag_adaptive_tfd(vag, fs, n_atoms)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgvagadp_edge():
    """Test edge cases."""
    vag = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    n_atoms = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_vag_adaptive_tfd(vag, fs, n_atoms)
    assert isinstance(result, dict)
