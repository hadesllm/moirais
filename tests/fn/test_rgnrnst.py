"""Tests for rgnrnst.rangayyan_nernst_potential."""
import numpy as np
import pytest
from morie.fn.rgnrnst import rangayyan_nernst_potential


def test_rgnrnst_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    conc_out = np.random.default_rng(42).normal(0, 1, 100)
    conc_in = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_nernst_potential(T, z, conc_out, conc_in)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgnrnst_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    conc_out = np.random.default_rng(42).normal(0, 1, 100)
    conc_in = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_nernst_potential(T, z, conc_out, conc_in)
    assert isinstance(result, dict)
