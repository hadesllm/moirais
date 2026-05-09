"""Tests for rgeemd.rangayyan_eemd."""
import numpy as np
import pytest
from moirais.fn.rgeemd import rangayyan_eemd


def test_rgeemd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n_ensembles = np.random.default_rng(42).normal(0, 1, 100)
    noise_std = np.random.default_rng(42).normal(0, 1, 100)
    max_imfs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_eemd(x, n_ensembles, noise_std, max_imfs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgeemd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n_ensembles = np.random.default_rng(42).normal(0, 1, 100)
    noise_std = np.random.default_rng(42).normal(0, 1, 100)
    max_imfs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_eemd(x, n_ensembles, noise_std, max_imfs)
    assert isinstance(result, dict)
