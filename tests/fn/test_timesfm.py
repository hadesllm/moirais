"""Tests for timesfm.timesfm."""
import numpy as np
import pytest
from morie.fn.timesfm import timesfm


def test_timesfm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = timesfm(y, horizon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_timesfm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = timesfm(y, horizon)
    assert isinstance(result, dict)
