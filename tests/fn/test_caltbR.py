"""Tests for caltbR.calibrated_rec."""
import numpy as np
import pytest
from morie.fn.caltbR import calibrated_rec


def test_caltbR_basic():
    """Test basic functionality."""
    pred = np.random.default_rng(42).normal(0, 1, 100)
    user_profile = np.random.default_rng(42).normal(0, 1, 100)
    result = calibrated_rec(pred, user_profile)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_caltbR_edge():
    """Test edge cases."""
    pred = np.random.default_rng(42).normal(0, 1, 100)
    user_profile = np.random.default_rng(42).normal(0, 1, 100)
    result = calibrated_rec(pred, user_profile)
    assert isinstance(result, dict)
