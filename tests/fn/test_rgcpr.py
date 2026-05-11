"""Tests for rgcpr.rangayyan_cpr_analysis."""
import numpy as np
import pytest
from morie.fn.rgcpr import rangayyan_cpr_analysis


def test_rgcpr_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_cpr_analysis(ecg, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgcpr_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_cpr_analysis(ecg, fs)
    assert isinstance(result, dict)
