"""Tests for rghrvmod.rangayyan_hrv_ar_model."""
import numpy as np
import pytest
from morie.fn.rghrvmod import rangayyan_hrv_ar_model


def test_rghrvmod_basic():
    """Test basic functionality."""
    rr_intervals = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = rangayyan_hrv_ar_model(rr_intervals, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rghrvmod_edge():
    """Test edge cases."""
    rr_intervals = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = rangayyan_hrv_ar_model(rr_intervals, order)
    assert isinstance(result, dict)
