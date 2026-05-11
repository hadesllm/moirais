"""Tests for sscompv.survival_competing_validation."""
import numpy as np
import pytest
from morie.fn.sscompv import survival_competing_validation


def test_sscompv_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event_type = np.random.default_rng(42).normal(0, 1, 100)
    predicted_F = np.random.default_rng(42).normal(0, 1, 100)
    result = survival_competing_validation(time, event_type, predicted_F)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sscompv_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event_type = np.random.default_rng(42).normal(0, 1, 100)
    predicted_F = np.random.default_rng(42).normal(0, 1, 100)
    result = survival_competing_validation(time, event_type, predicted_F)
    assert isinstance(result, dict)
