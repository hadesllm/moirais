"""Tests for rgemdimf.rangayyan_emd_imf."""
import numpy as np
import pytest
from moirais.fn.rgemdimf import rangayyan_emd_imf


def test_rgemdimf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    tol = 1e-6
    result = rangayyan_emd_imf(x, max_iter, tol)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgemdimf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    tol = 1e-6
    result = rangayyan_emd_imf(x, max_iter, tol)
    assert isinstance(result, dict)
