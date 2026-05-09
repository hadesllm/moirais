"""Tests for bigtm.bigram_topic."""
import numpy as np
import pytest
from moirais.fn.bigtm import bigram_topic


def test_bigtm_basic():
    """Test basic functionality."""
    docs = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = bigram_topic(docs, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bigtm_edge():
    """Test edge cases."""
    docs = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = bigram_topic(docs, K)
    assert isinstance(result, dict)
