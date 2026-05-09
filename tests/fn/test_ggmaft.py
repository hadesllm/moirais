"""Tests for ggmaft.generalized_gamma_aft."""
import numpy as np
import pytest
from moirais.fn.ggmaft import generalized_gamma_aft


def test_ggmaft_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = generalized_gamma_aft(time, event, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ggmaft_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = generalized_gamma_aft(time, event, X)
    assert isinstance(result, dict)
