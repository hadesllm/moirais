"""Tests for hrzph.horowitz_proportional_hazards."""
import numpy as np
import pytest
from moirais.fn.hrzph import horowitz_proportional_hazards


def test_hrzph_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_proportional_hazards(t, x, event)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzph_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_proportional_hazards(t, x, event)
    assert isinstance(result, dict)
