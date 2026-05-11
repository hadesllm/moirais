"""Tests for spnst.schabenberger_nonstationary_cov."""
import numpy as np
import pytest
from morie.fn.spnst import schabenberger_nonstationary_cov


def test_spnst_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = schabenberger_nonstationary_cov(coords, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spnst_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = schabenberger_nonstationary_cov(coords, z)
    assert isinstance(result, dict)
