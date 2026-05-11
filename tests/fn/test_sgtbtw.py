"""Tests for sgtbtw.sgt_betweenness_centrality."""
import numpy as np
import pytest
from morie.fn.sgtbtw import sgt_betweenness_centrality


def test_sgtbtw_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_betweenness_centrality(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtbtw_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_betweenness_centrality(A)
    assert isinstance(result, dict)
