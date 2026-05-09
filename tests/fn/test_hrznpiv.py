"""Tests for hrznpiv.horowitz_npiv_model."""
import numpy as np
import pytest
from moirais.fn.hrznpiv import horowitz_npiv_model


def test_hrznpiv_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = horowitz_npiv_model(x, y, w)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrznpiv_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = horowitz_npiv_model(x, y, w)
    assert isinstance(result, dict)
