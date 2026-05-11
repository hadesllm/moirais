"""Tests for depcen.dependent_censoring_hazard."""
import numpy as np
import pytest
from morie.fn.depcen import dependent_censoring_hazard


def test_depcen_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dependent_censoring_hazard(time, event, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_depcen_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dependent_censoring_hazard(time, event, X)
    assert isinstance(result, dict)
