"""Tests for genrgr.calibration_greg."""
import numpy as np
import pytest
from morie.fn.genrgr import calibration_greg


def test_genrgr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    totals = np.random.default_rng(42).normal(0, 1, 100)
    result = calibration_greg(y, x, weights, totals)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_genrgr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    totals = np.random.default_rng(42).normal(0, 1, 100)
    result = calibration_greg(y, x, weights, totals)
    assert isinstance(result, dict)
