"""Tests for survipw.ipcw_estimator."""
import numpy as np
import pytest
from morie.fn.survipw import ipcw_estimator


def test_survipw_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    cens_model = np.random.default_rng(42).normal(0, 1, 100)
    result = ipcw_estimator(time, event, cens_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survipw_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    cens_model = np.random.default_rng(42).normal(0, 1, 100)
    result = ipcw_estimator(time, event, cens_model)
    assert isinstance(result, dict)
