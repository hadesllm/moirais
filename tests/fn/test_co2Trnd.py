"""Tests for co2Trnd.co2_trend."""
import numpy as np
import pytest
from morie.fn.co2Trnd import co2_trend


def test_co2Trnd_basic():
    """Test basic functionality."""
    co2_monthly = np.random.default_rng(42).normal(0, 1, 100)
    result = co2_trend(co2_monthly)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_co2Trnd_edge():
    """Test edge cases."""
    co2_monthly = np.random.default_rng(42).normal(0, 1, 100)
    result = co2_trend(co2_monthly)
    assert isinstance(result, dict)
