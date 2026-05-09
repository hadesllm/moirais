"""Tests for rgderqrs.rangayyan_deriv_qrs."""
import numpy as np
import pytest
from moirais.fn.rgderqrs import rangayyan_deriv_qrs


def test_rgderqrs_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_deriv_qrs(ecg, fs, threshold)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgderqrs_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_deriv_qrs(ecg, fs, threshold)
    assert isinstance(result, dict)
