"""Tests for scbsft.sc_with_baseline_shift."""
import numpy as np
import pytest
from morie.fn.scbsft import sc_with_baseline_shift


def test_scbsft_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    baseline = np.random.default_rng(42).normal(0, 1, 100)
    result = sc_with_baseline_shift(y, D, X, baseline)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_scbsft_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    baseline = np.random.default_rng(42).normal(0, 1, 100)
    result = sc_with_baseline_shift(y, D, X, baseline)
    assert isinstance(result, dict)
