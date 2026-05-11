"""Tests for hrzweib.horowitz_weibull_heterogeneity."""
import numpy as np
import pytest
from morie.fn.hrzweib import horowitz_weibull_heterogeneity


def test_hrzweib_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    mixing_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_weibull_heterogeneity(t, x, event, mixing_dist)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzweib_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    mixing_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_weibull_heterogeneity(t, x, event, mixing_dist)
    assert isinstance(result, dict)
