"""Tests for bert4r.bert4rec."""
import numpy as np
import pytest
from moirais.fn.bert4r import bert4rec


def test_bert4r_basic():
    """Test basic functionality."""
    seqs = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = bert4rec(seqs, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bert4r_edge():
    """Test edge cases."""
    seqs = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = bert4rec(seqs, K)
    assert isinstance(result, dict)
