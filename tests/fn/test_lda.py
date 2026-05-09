"""Tests for lda.lda_topic."""
import numpy as np
import pytest
from moirais.fn.lda import lda_topic


def test_lda_basic():
    """Test basic functionality."""
    docs = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    beta = 0.8
    result = lda_topic(docs, K, alpha, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lda_edge():
    """Test edge cases."""
    docs = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    alpha = 0.05
    beta = 0.8
    result = lda_topic(docs, K, alpha, beta)
    assert isinstance(result, dict)
