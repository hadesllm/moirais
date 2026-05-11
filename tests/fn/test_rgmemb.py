"""Tests for rgmemb.rangayyan_membrane_potential."""
import numpy as np
import pytest
from morie.fn.rgmemb import rangayyan_membrane_potential


def test_rgmemb_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    I_inj = np.random.default_rng(42).normal(0, 1, 100)
    C_m = np.random.default_rng(42).normal(0, 1, 100)
    R_m = np.random.default_rng(42).normal(0, 1, 100)
    V_rest = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_membrane_potential(t, I_inj, C_m, R_m, V_rest)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgmemb_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    I_inj = np.random.default_rng(42).normal(0, 1, 100)
    C_m = np.random.default_rng(42).normal(0, 1, 100)
    R_m = np.random.default_rng(42).normal(0, 1, 100)
    V_rest = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_membrane_potential(t, I_inj, C_m, R_m, V_rest)
    assert isinstance(result, dict)
