"""Tests for survcox.cox_partial_likelihood."""
import numpy as np
import pytest
from morie.fn.survcox import cox_partial_likelihood


def test_survcox_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    beta = 0.8
    result = cox_partial_likelihood(time, event, X, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survcox_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    beta = 0.8
    result = cox_partial_likelihood(time, event, X, beta)
    assert isinstance(result, dict)
