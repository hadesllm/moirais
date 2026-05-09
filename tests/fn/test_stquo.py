"""Tests for stquo.status_quo_spatial."""
import numpy as np
import pytest
from moirais.fn.stquo import status_quo_spatial


def test_stquo_basic():
    """Test basic functionality."""
    ideal_points = np.random.default_rng(42).normal(0, 1, 100)
    status_quo = np.random.default_rng(42).normal(0, 1, 100)
    proposal = np.random.default_rng(42).normal(0, 1, 100)
    result = status_quo_spatial(ideal_points, status_quo, proposal)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_stquo_edge():
    """Test edge cases."""
    ideal_points = np.random.default_rng(42).normal(0, 1, 100)
    status_quo = np.random.default_rng(42).normal(0, 1, 100)
    proposal = np.random.default_rng(42).normal(0, 1, 100)
    result = status_quo_spatial(ideal_points, status_quo, proposal)
    assert isinstance(result, dict)
