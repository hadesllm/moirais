"""Tests for alfrf2.rfdiffusion_protein."""
import numpy as np
import pytest
from morie.fn.alfrf2 import rfdiffusion_protein


def test_alfrf2_basic():
    """Test basic functionality."""
    target_motif = np.random.default_rng(42).normal(0, 1, 100)
    scaffold = np.random.default_rng(42).normal(0, 1, 100)
    result = rfdiffusion_protein(target_motif, scaffold)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfrf2_edge():
    """Test edge cases."""
    target_motif = np.random.default_rng(42).normal(0, 1, 100)
    scaffold = np.random.default_rng(42).normal(0, 1, 100)
    result = rfdiffusion_protein(target_motif, scaffold)
    assert isinstance(result, dict)
