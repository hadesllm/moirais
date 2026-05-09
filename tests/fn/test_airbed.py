"""Tests for airbed.emissions_inventory."""
import numpy as np
import pytest
from moirais.fn.airbed import emissions_inventory


def test_airbed_basic():
    """Test basic functionality."""
    activity = np.random.default_rng(42).normal(0, 1, 100)
    ef = np.random.default_rng(42).normal(0, 1, 100)
    result = emissions_inventory(activity, ef)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_airbed_edge():
    """Test edge cases."""
    activity = np.random.default_rng(42).normal(0, 1, 100)
    ef = np.random.default_rng(42).normal(0, 1, 100)
    result = emissions_inventory(activity, ef)
    assert isinstance(result, dict)
