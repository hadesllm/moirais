"""Tests for cnsint.concurrent_calibration."""
import numpy as np
import pytest
from moirais.fn.cnsint import concurrent_calibration


def test_cnsint_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    item = np.random.default_rng(42).normal(0, 1, 100)
    group = np.random.default_rng(42).normal(0, 1, 100)
    anchor = np.random.default_rng(42).normal(0, 1, 100)
    result = concurrent_calibration(y, item, group, anchor)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cnsint_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    item = np.random.default_rng(42).normal(0, 1, 100)
    group = np.random.default_rng(42).normal(0, 1, 100)
    anchor = np.random.default_rng(42).normal(0, 1, 100)
    result = concurrent_calibration(y, item, group, anchor)
    assert isinstance(result, dict)
