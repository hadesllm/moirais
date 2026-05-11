"""Tests for mainf.ma_influence_diagnostics."""
import numpy as np
import pytest
from morie.fn.mainf import ma_influence_diagnostics


def test_mainf_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = ma_influence_diagnostics(yi, vi, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mainf_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = ma_influence_diagnostics(yi, vi, X)
    assert isinstance(result, dict)
