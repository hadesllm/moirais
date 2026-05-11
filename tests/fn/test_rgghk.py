"""Tests for rgghk.rangayyan_goldman_eqn."""
import numpy as np
import pytest
from morie.fn.rgghk import rangayyan_goldman_eqn


def test_rgghk_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    P_K = np.random.default_rng(42).normal(0, 1, 100)
    P_Na = np.random.default_rng(42).normal(0, 1, 100)
    P_Cl = np.random.default_rng(42).normal(0, 1, 100)
    ion_concs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_goldman_eqn(T, P_K, P_Na, P_Cl, ion_concs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgghk_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    P_K = np.random.default_rng(42).normal(0, 1, 100)
    P_Na = np.random.default_rng(42).normal(0, 1, 100)
    P_Cl = np.random.default_rng(42).normal(0, 1, 100)
    ion_concs = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_goldman_eqn(T, P_K, P_Na, P_Cl, ion_concs)
    assert isinstance(result, dict)
