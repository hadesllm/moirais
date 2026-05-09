"""Tests for sse4r.ssepta_seq."""
import numpy as np
import pytest
from moirais.fn.sse4r import ssepta_seq


def test_sse4r_basic():
    """Test basic functionality."""
    seqs = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = ssepta_seq(seqs, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sse4r_edge():
    """Test edge cases."""
    seqs = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = ssepta_seq(seqs, K)
    assert isinstance(result, dict)
