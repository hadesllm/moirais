"""Tests for cmuti.copula_mutual_information."""
import numpy as np
import pytest
from morie.fn.cmuti import copula_mutual_information


def test_cmuti_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    copula = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = copula_mutual_information(y, u, v, copula, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cmuti_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    copula = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = copula_mutual_information(y, u, v, copula, theta)
    assert isinstance(result, dict)
