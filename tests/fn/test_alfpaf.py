"""Tests for alfpaf.alphafold_pair_repr."""
import numpy as np
import pytest
from moirais.fn.alfpaf import alphafold_pair_repr


def test_alfpaf_basic():
    """Test basic functionality."""
    s = 90
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = alphafold_pair_repr(s, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfpaf_edge():
    """Test edge cases."""
    s = 90
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = alphafold_pair_repr(s, z)
    assert isinstance(result, dict)
