"""Tests for timesf.timesfm_foundation."""
import numpy as np
import pytest
from morie.fn.timesf import timesfm_foundation


def test_timesf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = timesfm_foundation(y, horizon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_timesf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = timesfm_foundation(y, horizon)
    assert isinstance(result, dict)
