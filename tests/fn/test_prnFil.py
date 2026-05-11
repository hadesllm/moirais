"""Tests for prnFil.prophet_changepoint."""
import numpy as np
import pytest
from morie.fn.prnFil import prophet_changepoint


def test_prnFil_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    n_cps = np.random.default_rng(42).normal(0, 1, 100)
    result = prophet_changepoint(y, n_cps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_prnFil_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    n_cps = np.random.default_rng(42).normal(0, 1, 100)
    result = prophet_changepoint(y, n_cps)
    assert isinstance(result, dict)
