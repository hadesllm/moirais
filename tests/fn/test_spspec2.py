"""Tests for spspec2.schabenberger_spectral_sim."""
import numpy as np
import pytest
from moirais.fn.spspec2 import schabenberger_spectral_sim


def test_spspec2_basic():
    """Test basic functionality."""
    spectral_density = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    n_freqs = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_spectral_sim(spectral_density, coords, n_freqs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spspec2_edge():
    """Test edge cases."""
    spectral_density = np.random.default_rng(42).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    n_freqs = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_spectral_sim(spectral_density, coords, n_freqs)
    assert isinstance(result, dict)
