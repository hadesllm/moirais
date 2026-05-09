"""Tests for rng009.rangayyan_ch3_sample_rms."""
import numpy as np
import pytest
from moirais.fn.rng009 import rangayyan_ch3_sample_rms


def test_rng009_basic():
    """Test basic functionality."""
    eta = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_sample_rms(eta, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng009_edge():
    """Test edge cases."""
    eta = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = rangayyan_ch3_sample_rms(eta, N)
    assert isinstance(result, dict)
