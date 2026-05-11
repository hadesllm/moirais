"""Tests for spkint.spike_information."""
import numpy as np
import pytest
from morie.fn.spkint import spike_information


def test_spkint_basic():
    """Test basic functionality."""
    spike = np.random.default_rng(42).normal(0, 1, 100)
    stim = np.random.default_rng(42).normal(0, 1, 100)
    result = spike_information(spike, stim)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spkint_edge():
    """Test edge cases."""
    spike = np.random.default_rng(42).normal(0, 1, 100)
    stim = np.random.default_rng(42).normal(0, 1, 100)
    result = spike_information(spike, stim)
    assert isinstance(result, dict)
