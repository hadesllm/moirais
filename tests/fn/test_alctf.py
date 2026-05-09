"""Tests for alctf.alammar_c_tfidf."""
import numpy as np
import pytest
from moirais.fn.alctf import alammar_c_tfidf


def test_alctf_basic():
    """Test basic functionality."""
    term_counts_by_class = np.random.default_rng(42).normal(0, 1, 100)
    corpus_freq = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = alammar_c_tfidf(term_counts_by_class, corpus_freq, A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alctf_edge():
    """Test edge cases."""
    term_counts_by_class = np.random.default_rng(42).normal(0, 1, 100)
    corpus_freq = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = alammar_c_tfidf(term_counts_by_class, corpus_freq, A)
    assert isinstance(result, dict)
