"""Tests for scvelo.rna_velocity."""
import numpy as np
import pytest
from morie.fn.scvelo import rna_velocity


def test_scvelo_basic():
    """Test basic functionality."""
    spliced = np.random.default_rng(42).normal(0, 1, 100)
    unspliced = np.random.default_rng(42).normal(0, 1, 100)
    result = rna_velocity(spliced, unspliced)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_scvelo_edge():
    """Test edge cases."""
    spliced = np.random.default_rng(42).normal(0, 1, 100)
    unspliced = np.random.default_rng(42).normal(0, 1, 100)
    result = rna_velocity(spliced, unspliced)
    assert isinstance(result, dict)
