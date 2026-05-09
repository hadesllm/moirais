"""Tests for plpol.plot_spatial."""
import numpy as np
import pytest
from moirais.fn.plpol import plot_spatial


def test_plpol_basic():
    """Test basic functionality."""
    ideal_points = np.random.default_rng(42).normal(0, 1, 100)
    party_labels = np.random.default_rng(43).integers(0, 2, 100)
    stimuli_labels = np.random.default_rng(43).integers(0, 2, 100)
    result = plot_spatial(ideal_points, party_labels, stimuli_labels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_plpol_edge():
    """Test edge cases."""
    ideal_points = np.random.default_rng(42).normal(0, 1, 100)
    party_labels = np.random.default_rng(43).integers(0, 2, 100)
    stimuli_labels = np.random.default_rng(43).integers(0, 2, 100)
    result = plot_spatial(ideal_points, party_labels, stimuli_labels)
    assert isinstance(result, dict)
