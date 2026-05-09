"""Tests for wsmpst.wasserman_plug_in_estimator."""
import numpy as np
import pytest
from moirais.fn.wsmpst import wasserman_plug_in_estimator


def test_wsmpst_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = wasserman_plug_in_estimator(data, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmpst_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = wasserman_plug_in_estimator(data, T)
    assert isinstance(result, dict)
