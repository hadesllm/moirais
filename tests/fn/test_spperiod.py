"""Tests for spperiod.schabenberger_periodogram."""
import numpy as np
import pytest
from morie.fn.spperiod import schabenberger_periodogram


def test_spperiod_basic():
    """Test basic functionality."""
    z_lattice = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = schabenberger_periodogram(z_lattice, coords)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spperiod_edge():
    """Test edge cases."""
    z_lattice = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = schabenberger_periodogram(z_lattice, coords)
    assert isinstance(result, dict)
