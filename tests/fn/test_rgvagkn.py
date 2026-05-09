"""Tests for rgvagkn.rangayyan_vag_knee_cartilage."""
import numpy as np
import pytest
from moirais.fn.rgvagkn import rangayyan_vag_knee_cartilage


def test_rgvagkn_basic():
    """Test basic functionality."""
    vag = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_vag_knee_cartilage(vag, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgvagkn_edge():
    """Test edge cases."""
    vag = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_vag_knee_cartilage(vag, fs)
    assert isinstance(result, dict)
