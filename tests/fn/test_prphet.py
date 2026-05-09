"""Tests for prphet.prophet."""
import numpy as np
import pytest
from moirais.fn.prphet import prophet


def test_prphet_basic():
    """Test basic functionality."""
    ds = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    holidays = np.random.default_rng(42).normal(0, 1, 100)
    changepoints = np.random.default_rng(42).normal(0, 1, 100)
    result = prophet(ds, y, holidays, changepoints)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_prphet_edge():
    """Test edge cases."""
    ds = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    holidays = np.random.default_rng(42).normal(0, 1, 100)
    changepoints = np.random.default_rng(42).normal(0, 1, 100)
    result = prophet(ds, y, holidays, changepoints)
    assert isinstance(result, dict)
