"""Tests for alfevf.alphafold_evoformer."""
import numpy as np
import pytest
from moirais.fn.alfevf import alphafold_evoformer


def test_alfevf_basic():
    """Test basic functionality."""
    s = 90
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = alphafold_evoformer(s, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfevf_edge():
    """Test edge cases."""
    s = 90
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = alphafold_evoformer(s, z)
    assert isinstance(result, dict)
