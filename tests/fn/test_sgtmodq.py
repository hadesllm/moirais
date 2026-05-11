"""Tests for sgtmodq.sgt_modularity_q."""
import numpy as np
import pytest
from morie.fn.sgtmodq import sgt_modularity_q


def test_sgtmodq_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    labels = np.random.default_rng(43).integers(0, 2, 100)
    result = sgt_modularity_q(A, labels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtmodq_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    labels = np.random.default_rng(43).integers(0, 2, 100)
    result = sgt_modularity_q(A, labels)
    assert isinstance(result, dict)
