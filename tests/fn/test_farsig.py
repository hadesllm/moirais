"""Tests for farsig.farrington_signal."""
import numpy as np
import pytest
from morie.fn.farsig import farrington_signal


def test_farsig_basic():
    """Test basic functionality."""
    counts = np.random.default_rng(42).normal(0, 1, 100)
    baseline_years = np.random.default_rng(42).normal(0, 1, 100)
    reference_window = np.random.default_rng(42).normal(0, 1, 100)
    result = farrington_signal(counts, baseline_years, reference_window)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_farsig_edge():
    """Test edge cases."""
    counts = np.random.default_rng(42).normal(0, 1, 100)
    baseline_years = np.random.default_rng(42).normal(0, 1, 100)
    reference_window = np.random.default_rng(42).normal(0, 1, 100)
    result = farrington_signal(counts, baseline_years, reference_window)
    assert isinstance(result, dict)
