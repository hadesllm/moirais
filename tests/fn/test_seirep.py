"""Tests for seirep.seir_compartmental."""
import numpy as np
import pytest
from morie.fn.seirep import seir_compartmental


def test_seirep_basic():
    """Test basic functionality."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    I = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    sigma = 1.0
    gamma = 1.0
    result = seir_compartmental(S, E, I, R, beta, sigma, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_seirep_edge():
    """Test edge cases."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    I = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    sigma = 1.0
    gamma = 1.0
    result = seir_compartmental(S, E, I, R, beta, sigma, gamma)
    assert isinstance(result, dict)
