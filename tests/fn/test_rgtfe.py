"""Tests for rgtfe.rangayyan_transfer_func_est."""
import numpy as np
import pytest
from moirais.fn.rgtfe import rangayyan_transfer_func_est


def test_rgtfe_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    fs = 100.0
    nperseg = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_transfer_func_est(x, y, fs, nperseg)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgtfe_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    fs = 100.0
    nperseg = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_transfer_func_est(x, y, fs, nperseg)
    assert isinstance(result, dict)
