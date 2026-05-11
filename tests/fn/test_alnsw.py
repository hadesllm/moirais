"""Tests for alnsw.smith_waterman."""
import numpy as np
import pytest
from morie.fn.alnsw import smith_waterman


def test_alnsw_basic():
    """Test basic functionality."""
    seq1 = np.random.default_rng(42).normal(0, 1, 100)
    seq2 = np.random.default_rng(42).normal(0, 1, 100)
    sub_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    gap = np.random.default_rng(42).normal(0, 1, 100)
    result = smith_waterman(seq1, seq2, sub_matrix, gap)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alnsw_edge():
    """Test edge cases."""
    seq1 = np.random.default_rng(42).normal(0, 1, 100)
    seq2 = np.random.default_rng(42).normal(0, 1, 100)
    sub_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    gap = np.random.default_rng(42).normal(0, 1, 100)
    result = smith_waterman(seq1, seq2, sub_matrix, gap)
    assert isinstance(result, dict)
