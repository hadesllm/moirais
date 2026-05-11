"""Tests for coxtmv.cox_time_varying."""
import numpy as np
import pytest
from morie.fn.coxtmv import cox_time_varying


def test_coxtmv_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X_time = np.random.default_rng(42).normal(0, 1, 100)
    result = cox_time_varying(time, event, X_time)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_coxtmv_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X_time = np.random.default_rng(42).normal(0, 1, 100)
    result = cox_time_varying(time, event, X_time)
    assert isinstance(result, dict)
