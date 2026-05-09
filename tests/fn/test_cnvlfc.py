"""Tests for cnvlfc.convergent_cross_mapping."""
import numpy as np
import pytest
from moirais.fn.cnvlfc import convergent_cross_mapping


def test_cnvlfc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = convergent_cross_mapping(x, y, E, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cnvlfc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = convergent_cross_mapping(x, y, E, tau)
    assert isinstance(result, dict)
