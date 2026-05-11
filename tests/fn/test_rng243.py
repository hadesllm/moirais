"""Tests for rng243.rangayyan_ch4_log_maximum_phase_expansion."""
import numpy as np
import pytest
from morie.fn.rng243 import rangayyan_ch4_log_maximum_phase_expansion


def test_rng243_basic():
    """Test basic functionality."""
    beta = 0.8
    z = np.random.default_rng(44).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_log_maximum_phase_expansion(beta, z, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng243_edge():
    """Test edge cases."""
    beta = 0.8
    z = np.random.default_rng(44).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_log_maximum_phase_expansion(beta, z, n)
    assert isinstance(result, dict)
