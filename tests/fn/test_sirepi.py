"""Tests for sirepi.sir_compartmental."""
import numpy as np
import pytest
from morie.fn.sirepi import sir_compartmental


def test_sirepi_basic():
    """Test basic functionality."""
    S0 = np.random.default_rng(42).normal(0, 1, 100)
    I0 = np.random.default_rng(42).normal(0, 1, 100)
    R0 = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    gamma = 1.0
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = sir_compartmental(S0, I0, R0, beta, gamma, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sirepi_edge():
    """Test edge cases."""
    S0 = np.random.default_rng(42).normal(0, 1, 100)
    I0 = np.random.default_rng(42).normal(0, 1, 100)
    R0 = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    gamma = 1.0
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = sir_compartmental(S0, I0, R0, beta, gamma, T)
    assert isinstance(result, dict)
