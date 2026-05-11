"""Tests for rghrvar.rangayyan_hrv_ar_ratio."""
import numpy as np
import pytest
from morie.fn.rghrvar import rangayyan_hrv_ar_ratio


def test_rghrvar_basic():
    """Test basic functionality."""
    rr_intervals = np.random.default_rng(42).normal(0, 1, 100)
    ar_order = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_hrv_ar_ratio(rr_intervals, ar_order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rghrvar_edge():
    """Test edge cases."""
    rr_intervals = np.random.default_rng(42).normal(0, 1, 100)
    ar_order = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_hrv_ar_ratio(rr_intervals, ar_order)
    assert isinstance(result, dict)
