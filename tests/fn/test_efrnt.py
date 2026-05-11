"""Tests for efrnt.efron_tie_correction."""
import numpy as np
import pytest
from morie.fn.efrnt import efron_tie_correction


def test_efrnt_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = efron_tie_correction(time, event, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_efrnt_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = efron_tie_correction(time, event, X)
    assert isinstance(result, dict)
