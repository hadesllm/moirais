"""Tests for hrzd1.horowitz_duration_model."""
import numpy as np
import pytest
from moirais.fn.hrzd1 import horowitz_duration_model


def test_hrzd1_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_duration_model(t, x, event)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzd1_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_duration_model(t, x, event)
    assert isinstance(result, dict)
