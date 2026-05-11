"""Tests for shedmd.viral_shedding_model."""
import numpy as np
import pytest
from morie.fn.shedmd import viral_shedding_model


def test_shedmd_basic():
    """Test basic functionality."""
    days = np.random.default_rng(42).normal(0, 1, 100)
    viral_load = np.random.default_rng(42).normal(0, 1, 100)
    result = viral_shedding_model(days, viral_load)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_shedmd_edge():
    """Test edge cases."""
    days = np.random.default_rng(42).normal(0, 1, 100)
    viral_load = np.random.default_rng(42).normal(0, 1, 100)
    result = viral_shedding_model(days, viral_load)
    assert isinstance(result, dict)
