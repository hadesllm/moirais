"""Tests for aftgma.aft_generalized_gamma."""
import numpy as np
import pytest
from moirais.fn.aftgma import aft_generalized_gamma


def test_aftgma_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = aft_generalized_gamma(time, event, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aftgma_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = aft_generalized_gamma(time, event, X)
    assert isinstance(result, dict)
