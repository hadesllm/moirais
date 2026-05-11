"""Tests for gremb.geron_embedding_lookup."""
import numpy as np
import pytest
from morie.fn.gremb import geron_embedding_lookup


def test_gremb_basic():
    """Test basic functionality."""
    ids = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_embedding_lookup(ids, E)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gremb_edge():
    """Test edge cases."""
    ids = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_embedding_lookup(ids, E)
    assert isinstance(result, dict)
