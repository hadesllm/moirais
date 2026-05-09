"""Tests for boryis.borusyak_jaravel_spiess."""
import numpy as np
import pytest
from moirais.fn.boryis import borusyak_jaravel_spiess


def test_boryis_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = borusyak_jaravel_spiess(y, D, unit, time, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_boryis_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = borusyak_jaravel_spiess(y, D, unit, time, X)
    assert isinstance(result, dict)
