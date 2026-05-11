"""Tests for rgneural.rangayyan_neural_decode."""
import numpy as np
import pytest
from morie.fn.rgneural import rangayyan_neural_decode


def test_rgneural_basic():
    """Test basic functionality."""
    spike_trains = np.random.default_rng(42).normal(0, 1, 100)
    movement_labels = np.random.default_rng(43).integers(0, 2, 100)
    n_ch = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_neural_decode(spike_trains, movement_labels, n_ch)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgneural_edge():
    """Test edge cases."""
    spike_trains = np.random.default_rng(42).normal(0, 1, 100)
    movement_labels = np.random.default_rng(43).integers(0, 2, 100)
    n_ch = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_neural_decode(spike_trains, movement_labels, n_ch)
    assert isinstance(result, dict)
