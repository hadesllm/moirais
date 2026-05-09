"""Tests for sgtclo.sgt_closeness_centrality."""
import numpy as np
import pytest
from moirais.fn.sgtclo import sgt_closeness_centrality


def test_sgtclo_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_closeness_centrality(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtclo_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_closeness_centrality(A)
    assert isinstance(result, dict)
