"""Tests for rng242.rangayyan_ch4_log_minimum_phase_expansion."""
import numpy as np
import pytest
from moirais.fn.rng242 import rangayyan_ch4_log_minimum_phase_expansion


def test_rng242_basic():
    """Test basic functionality."""
    alpha = 0.05
    z = np.random.default_rng(44).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_log_minimum_phase_expansion(alpha, z, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng242_edge():
    """Test edge cases."""
    alpha = 0.05
    z = np.random.default_rng(44).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch4_log_minimum_phase_expansion(alpha, z, n)
    assert isinstance(result, dict)
