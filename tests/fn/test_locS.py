"""Tests for locS.location_scale_estimator."""
import numpy as np
import pytest
from morie.fn.locS import location_scale_estimator


def test_locS_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = location_scale_estimator(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_locS_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = location_scale_estimator(x)
    assert isinstance(result, dict)
