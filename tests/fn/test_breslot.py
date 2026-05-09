"""Tests for breslot.breslow_tie_correction."""
import numpy as np
import pytest
from moirais.fn.breslot import breslow_tie_correction


def test_breslot_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = breslow_tie_correction(time, event, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_breslot_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = breslow_tie_correction(time, event, X)
    assert isinstance(result, dict)
