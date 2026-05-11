"""Tests for alfpea.alphafold_pae_predict."""
import numpy as np
import pytest
from morie.fn.alfpea import alphafold_pae_predict


def test_alfpea_basic():
    """Test basic functionality."""
    s = 90
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = alphafold_pae_predict(s, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfpea_edge():
    """Test edge cases."""
    s = 90
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = alphafold_pae_predict(s, z)
    assert isinstance(result, dict)
