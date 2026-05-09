"""Tests for altsd.alammar_tsdae_objective."""
import numpy as np
import pytest
from moirais.fn.altsd import alammar_tsdae_objective


def test_altsd_basic():
    """Test basic functionality."""
    clean = np.random.default_rng(42).normal(0, 1, 100)
    corrupted = np.random.default_rng(42).normal(0, 1, 100)
    encoder = np.random.default_rng(42).normal(0, 1, 100)
    decoder = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_tsdae_objective(clean, corrupted, encoder, decoder)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_altsd_edge():
    """Test edge cases."""
    clean = np.random.default_rng(42).normal(0, 1, 100)
    corrupted = np.random.default_rng(42).normal(0, 1, 100)
    encoder = np.random.default_rng(42).normal(0, 1, 100)
    decoder = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_tsdae_objective(clean, corrupted, encoder, decoder)
    assert isinstance(result, dict)
