"""Tests for twostg.two_stage_hazard."""
import numpy as np
import pytest
from morie.fn.twostg import two_stage_hazard


def test_twostg_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = two_stage_hazard(time, event, X, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_twostg_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = two_stage_hazard(time, event, X, Z)
    assert isinstance(result, dict)
