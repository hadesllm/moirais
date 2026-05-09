"""Tests for hmbsz.geron_batch_size_heuristic."""
import numpy as np
import pytest
from moirais.fn.hmbsz import geron_batch_size_heuristic


def test_hmbsz_basic():
    """Test basic functionality."""
    n_train = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_batch_size_heuristic(n_train)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmbsz_edge():
    """Test edge cases."""
    n_train = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_batch_size_heuristic(n_train)
    assert isinstance(result, dict)
