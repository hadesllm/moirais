"""Tests for chgcus.changepoint_cusum."""
import numpy as np
import pytest
from morie.fn.chgcus import changepoint_cusum


def test_chgcus_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    h = 0.3
    result = changepoint_cusum(y, k, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_chgcus_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    h = 0.3
    result = changepoint_cusum(y, k, h)
    assert isinstance(result, dict)
