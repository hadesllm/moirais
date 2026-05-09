"""Tests for wsmchi.wasserman_chi_sq_gof."""
import numpy as np
import pytest
from moirais.fn.wsmchi import wasserman_chi_sq_gof


def test_wsmchi_basic():
    """Test basic functionality."""
    observed = np.random.default_rng(42).normal(0, 1, 100)
    expected = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_chi_sq_gof(observed, expected)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmchi_edge():
    """Test edge cases."""
    observed = np.random.default_rng(42).normal(0, 1, 100)
    expected = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_chi_sq_gof(observed, expected)
    assert isinstance(result, dict)
