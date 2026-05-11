"""Tests for klmflt.kalman_filter."""
import numpy as np
import pytest
from morie.fn.klmflt import kalman_filter


def test_klmflt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kalman_filter(y, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_klmflt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kalman_filter(y, model)
    assert isinstance(result, dict)
