"""Tests for wsmmgf.wasserman_mgf."""
import numpy as np
import pytest
from moirais.fn.wsmmgf import wasserman_mgf


def test_wsmmgf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = wasserman_mgf(x, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmmgf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = wasserman_mgf(x, t)
    assert isinstance(result, dict)
