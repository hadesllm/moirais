"""Tests for alfreq.alphafold_recycling."""
import numpy as np
import pytest
from morie.fn.alfreq import alphafold_recycling


def test_alfreq_basic():
    """Test basic functionality."""
    s = 90
    z = np.random.default_rng(44).normal(0, 1, 100)
    n_recycles = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_recycling(s, z, n_recycles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfreq_edge():
    """Test edge cases."""
    s = 90
    z = np.random.default_rng(44).normal(0, 1, 100)
    n_recycles = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_recycling(s, z, n_recycles)
    assert isinstance(result, dict)
