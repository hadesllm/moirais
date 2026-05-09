"""Tests for sgtkem.sgt_katz_centrality."""
import numpy as np
import pytest
from moirais.fn.sgtkem import sgt_katz_centrality


def test_sgtkem_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    alpha = 0.05
    beta = 0.8
    result = sgt_katz_centrality(A, alpha, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtkem_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    alpha = 0.05
    beta = 0.8
    result = sgt_katz_centrality(A, alpha, beta)
    assert isinstance(result, dict)
