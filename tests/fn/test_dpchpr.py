"""Tests for dpchpr.dp_changepoint."""
import numpy as np
import pytest
from moirais.fn.dpchpr import dp_changepoint


def test_dpchpr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = dp_changepoint(y, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpchpr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    result = dp_changepoint(y, alpha)
    assert isinstance(result, dict)
