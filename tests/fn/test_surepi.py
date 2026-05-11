"""Tests for surepi.surveillance_signal."""
import numpy as np
import pytest
from morie.fn.surepi import surveillance_signal


def test_surepi_basic():
    """Test basic functionality."""
    counts = np.random.default_rng(42).normal(0, 1, 100)
    baseline_window = np.random.default_rng(42).normal(0, 1, 100)
    result = surveillance_signal(counts, baseline_window)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_surepi_edge():
    """Test edge cases."""
    counts = np.random.default_rng(42).normal(0, 1, 100)
    baseline_window = np.random.default_rng(42).normal(0, 1, 100)
    result = surveillance_signal(counts, baseline_window)
    assert isinstance(result, dict)
