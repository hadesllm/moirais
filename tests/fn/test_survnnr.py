"""Tests for survnnr.survival_neural_net."""
import numpy as np
import pytest
from moirais.fn.survnnr import survival_neural_net


def test_survnnr_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = survival_neural_net(time, event, X, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survnnr_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = survival_neural_net(time, event, X, model)
    assert isinstance(result, dict)
