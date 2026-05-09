"""Tests for bats.bats."""
import numpy as np
import pytest
from moirais.fn.bats import bats


def test_bats_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    seasonal_periods = np.random.default_rng(42).normal(0, 1, 100)
    result = bats(y, seasonal_periods)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bats_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    seasonal_periods = np.random.default_rng(42).normal(0, 1, 100)
    result = bats(y, seasonal_periods)
    assert isinstance(result, dict)
