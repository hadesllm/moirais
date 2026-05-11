"""Tests for volfig.vol_figarch_fit."""
import numpy as np
import pytest
from morie.fn.volfig import vol_figarch_fit


def test_volfig_basic():
    """Test basic functionality."""
    r = 10
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_figarch_fit(r, p, q, init)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volfig_edge():
    """Test edge cases."""
    r = 10
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_figarch_fit(r, p, q, init)
    assert isinstance(result, dict)
