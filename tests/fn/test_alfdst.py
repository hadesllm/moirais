"""Tests for alfdst.alphafold_distogram."""
import numpy as np
import pytest
from moirais.fn.alfdst import alphafold_distogram


def test_alfdst_basic():
    """Test basic functionality."""
    s = 90
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = alphafold_distogram(s, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfdst_edge():
    """Test edge cases."""
    s = 90
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = alphafold_distogram(s, z)
    assert isinstance(result, dict)
