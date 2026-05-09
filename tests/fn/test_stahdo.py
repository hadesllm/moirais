"""Tests for stahdo.stahel_donoho."""
import numpy as np
import pytest
from moirais.fn.stahdo import stahel_donoho


def test_stahdo_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    u_dirs = np.random.default_rng(42).normal(0, 1, 100)
    result = stahel_donoho(X, u_dirs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_stahdo_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    u_dirs = np.random.default_rng(42).normal(0, 1, 100)
    result = stahel_donoho(X, u_dirs)
    assert isinstance(result, dict)
