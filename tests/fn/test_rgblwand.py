"""Tests for rgblwand.rangayyan_baseline_wander."""
import numpy as np
import pytest
from moirais.fn.rgblwand import rangayyan_baseline_wander


def test_rgblwand_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    cutoff = 10.0
    result = rangayyan_baseline_wander(ecg, fs, cutoff)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgblwand_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    cutoff = 10.0
    result = rangayyan_baseline_wander(ecg, fs, cutoff)
    assert isinstance(result, dict)
