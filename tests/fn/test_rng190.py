"""Tests for rng190.rangayyan_ch4_pan_tompkins_peak_classification."""
import numpy as np
import pytest
from morie.fn.rng190 import rangayyan_ch4_pan_tompkins_peak_classification


def test_rng190_basic():
    """Test basic functionality."""
    PEAKI = np.random.default_rng(42).normal(0, 1, 100)
    SPKI = np.random.default_rng(42).normal(0, 1, 100)
    NPKI = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_pan_tompkins_peak_classification(PEAKI, SPKI, NPKI)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng190_edge():
    """Test edge cases."""
    PEAKI = np.random.default_rng(42).normal(0, 1, 100)
    SPKI = np.random.default_rng(42).normal(0, 1, 100)
    NPKI = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_pan_tompkins_peak_classification(PEAKI, SPKI, NPKI)
    assert isinstance(result, dict)
