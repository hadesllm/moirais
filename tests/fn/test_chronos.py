"""Tests for chronos.chronos_foundation_ts."""
import numpy as np
import pytest
from moirais.fn.chronos import chronos_foundation_ts


def test_chronos_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = chronos_foundation_ts(y, horizon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_chronos_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = chronos_foundation_ts(y, horizon)
    assert isinstance(result, dict)
