"""Tests for survbri.brier_score."""
import numpy as np
import pytest
from moirais.fn.survbri import brier_score


def test_survbri_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    predicted_S = np.random.default_rng(42).normal(0, 1, 100)
    t_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = brier_score(time, event, predicted_S, t_grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survbri_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    predicted_S = np.random.default_rng(42).normal(0, 1, 100)
    t_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = brier_score(time, event, predicted_S, t_grid)
    assert isinstance(result, dict)
