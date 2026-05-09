"""Tests for alfcnf.alphafold_confidence."""
import numpy as np
import pytest
from moirais.fn.alfcnf import alphafold_confidence


def test_alfcnf_basic():
    """Test basic functionality."""
    frames = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    result = alphafold_confidence(frames, s)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfcnf_edge():
    """Test edge cases."""
    frames = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    result = alphafold_confidence(frames, s)
    assert isinstance(result, dict)
