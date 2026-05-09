"""Tests for giss.giss_anomaly."""
import numpy as np
import pytest
from moirais.fn.giss import giss_anomaly


def test_giss_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    baseline = np.random.default_rng(42).normal(0, 1, 100)
    result = giss_anomaly(T, baseline)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_giss_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    baseline = np.random.default_rng(42).normal(0, 1, 100)
    result = giss_anomaly(T, baseline)
    assert isinstance(result, dict)
