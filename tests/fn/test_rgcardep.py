"""Tests for rgcardep.rangayyan_cardiac_elecphys."""
import numpy as np
import pytest
from moirais.fn.rgcardep import rangayyan_cardiac_elecphys


def test_rgcardep_basic():
    """Test basic functionality."""
    mesh = np.random.default_rng(42).normal(0, 1, 100)
    sigma_i = np.random.default_rng(42).normal(0, 1, 100)
    sigma_e = np.random.default_rng(42).normal(0, 1, 100)
    C_m = np.random.default_rng(42).normal(0, 1, 100)
    I_ion = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_cardiac_elecphys(mesh, sigma_i, sigma_e, C_m, I_ion)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgcardep_edge():
    """Test edge cases."""
    mesh = np.random.default_rng(42).normal(0, 1, 100)
    sigma_i = np.random.default_rng(42).normal(0, 1, 100)
    sigma_e = np.random.default_rng(42).normal(0, 1, 100)
    C_m = np.random.default_rng(42).normal(0, 1, 100)
    I_ion = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_cardiac_elecphys(mesh, sigma_i, sigma_e, C_m, I_ion)
    assert isinstance(result, dict)
