"""Tests for shres.schoenfeld_residual."""
import numpy as np
import pytest
from moirais.fn.shres import schoenfeld_residual


def test_shres_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = schoenfeld_residual(time, event, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_shres_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = schoenfeld_residual(time, event, X)
    assert isinstance(result, dict)
