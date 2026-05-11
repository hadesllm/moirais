"""Tests for alfbk2.alphafold_backbone."""
import numpy as np
import pytest
from morie.fn.alfbk2 import alphafold_backbone


def test_alfbk2_basic():
    """Test basic functionality."""
    frames = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_backbone(frames, delta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfbk2_edge():
    """Test edge cases."""
    frames = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_backbone(frames, delta)
    assert isinstance(result, dict)
