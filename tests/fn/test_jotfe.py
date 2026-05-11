"""Tests for jotfe.joseph_calendar_features."""
import numpy as np
import pytest
from morie.fn.jotfe import joseph_calendar_features


def test_jotfe_basic():
    """Test basic functionality."""
    timestamps = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_calendar_features(timestamps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jotfe_edge():
    """Test edge cases."""
    timestamps = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_calendar_features(timestamps)
    assert isinstance(result, dict)
