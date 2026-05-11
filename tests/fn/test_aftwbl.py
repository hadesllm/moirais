"""Tests for aftwbl.aft_weibull."""
import numpy as np
import pytest
from morie.fn.aftwbl import aft_weibull


def test_aftwbl_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = aft_weibull(time, event, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aftwbl_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = aft_weibull(time, event, X)
    assert isinstance(result, dict)
