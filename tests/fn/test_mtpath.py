"""Tests for mtpath.meta_path."""
import numpy as np
import pytest
from moirais.fn.mtpath import meta_path


def test_mtpath_basic():
    """Test basic functionality."""
    G = np.eye(10)
    node_types = np.random.default_rng(42).normal(0, 1, 100)
    metapath = np.random.default_rng(42).normal(0, 1, 100)
    result = meta_path(G, node_types, metapath)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mtpath_edge():
    """Test edge cases."""
    G = np.eye(10)
    node_types = np.random.default_rng(42).normal(0, 1, 100)
    metapath = np.random.default_rng(42).normal(0, 1, 100)
    result = meta_path(G, node_types, metapath)
    assert isinstance(result, dict)
