"""Tests for alfsts.alphafold_structure_transition."""
import numpy as np
import pytest
from moirais.fn.alfsts import alphafold_structure_transition


def test_alfsts_basic():
    """Test basic functionality."""
    s = 90
    z = np.random.default_rng(44).normal(0, 1, 100)
    frames = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_structure_transition(s, z, frames)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfsts_edge():
    """Test edge cases."""
    s = 90
    z = np.random.default_rng(44).normal(0, 1, 100)
    frames = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_structure_transition(s, z, frames)
    assert isinstance(result, dict)
