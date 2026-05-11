"""Tests for siaepi.sir_epidemic."""
import numpy as np
import pytest
from morie.fn.siaepi import sir_epidemic


def test_siaepi_basic():
    """Test basic functionality."""
    G = np.eye(10)
    beta = 0.8
    gamma = 1.0
    initial = np.random.default_rng(42).normal(0, 1, 100)
    result = sir_epidemic(G, beta, gamma, initial)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_siaepi_edge():
    """Test edge cases."""
    G = np.eye(10)
    beta = 0.8
    gamma = 1.0
    initial = np.random.default_rng(42).normal(0, 1, 100)
    result = sir_epidemic(G, beta, gamma, initial)
    assert isinstance(result, dict)
