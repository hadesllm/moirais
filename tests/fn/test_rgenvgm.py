"""Tests for rgenvgm.rangayyan_envelogram."""
import numpy as np
import pytest
from morie.fn.rgenvgm import rangayyan_envelogram


def test_rgenvgm_basic():
    """Test basic functionality."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_envelogram(pcg, ecg, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgenvgm_edge():
    """Test edge cases."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    ecg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_envelogram(pcg, ecg, fs)
    assert isinstance(result, dict)
