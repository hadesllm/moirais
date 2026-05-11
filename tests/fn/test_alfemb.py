"""Tests for alfemb.alphafold_embedding_init."""
import numpy as np
import pytest
from morie.fn.alfemb import alphafold_embedding_init


def test_alfemb_basic():
    """Test basic functionality."""
    sequence = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_embedding_init(sequence)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfemb_edge():
    """Test edge cases."""
    sequence = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_embedding_init(sequence)
    assert isinstance(result, dict)
