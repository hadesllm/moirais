"""Tests for rgpcg.rangayyan_pcg_segments."""
import numpy as np
import pytest
from morie.fn.rgpcg import rangayyan_pcg_segments


def test_rgpcg_basic():
    """Test basic functionality."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_pcg_segments(pcg, ecg, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgpcg_edge():
    """Test edge cases."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_pcg_segments(pcg, ecg, fs)
    assert isinstance(result, dict)
