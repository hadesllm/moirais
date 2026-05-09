"""Tests for driftF.drift_forecast."""
import numpy as np
import pytest
from moirais.fn.driftF import drift_forecast


def test_driftF_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    h = 0.3
    result = drift_forecast(y, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_driftF_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    h = 0.3
    result = drift_forecast(y, h)
    assert isinstance(result, dict)
