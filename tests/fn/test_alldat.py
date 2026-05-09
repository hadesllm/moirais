"""Tests for alldat.alammar_lda_topic_distribution."""
import numpy as np
import pytest
from moirais.fn.alldat import alammar_lda_topic_distribution


def test_alldat_basic():
    """Test basic functionality."""
    alpha = 0.05
    beta = 0.8
    n_topics = np.random.default_rng(42).normal(0, 1, 100)
    documents = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_lda_topic_distribution(alpha, beta, n_topics, documents)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alldat_edge():
    """Test edge cases."""
    alpha = 0.05
    beta = 0.8
    n_topics = np.random.default_rng(42).normal(0, 1, 100)
    documents = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_lda_topic_distribution(alpha, beta, n_topics, documents)
    assert isinstance(result, dict)
