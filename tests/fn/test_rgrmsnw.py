"""Tests for rgrmsnw.rangayyan_rms_noise."""
import numpy as np
import pytest
from moirais.fn.rgrmsnw import rangayyan_rms_noise


def test_rgrmsnw_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    noise_segments = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_rms_noise(x, noise_segments)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgrmsnw_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    noise_segments = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_rms_noise(x, noise_segments)
    assert isinstance(result, dict)
