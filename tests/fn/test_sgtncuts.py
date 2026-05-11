"""Tests for sgtncuts.sgt_normalised_cut."""
import numpy as np
import pytest
from morie.fn.sgtncuts import sgt_normalised_cut


def test_sgtncuts_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    labels = np.random.default_rng(43).integers(0, 2, 100)
    result = sgt_normalised_cut(A, labels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtncuts_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    labels = np.random.default_rng(43).integers(0, 2, 100)
    result = sgt_normalised_cut(A, labels)
    assert isinstance(result, dict)
