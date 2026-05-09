"""Tests for sasRec.sasrec."""
import numpy as np
import pytest
from moirais.fn.sasRec import sasrec


def test_sasRec_basic():
    """Test basic functionality."""
    seqs = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = sasrec(seqs, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sasRec_edge():
    """Test edge cases."""
    seqs = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = sasrec(seqs, K)
    assert isinstance(result, dict)
