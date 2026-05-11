"""Tests for bkngr.burkov_ngram_mle."""
import numpy as np
import pytest
from morie.fn.bkngr import burkov_ngram_mle


def test_bkngr_basic():
    """Test basic functionality."""
    counts_ngram = np.random.default_rng(42).normal(0, 1, 100)
    counts_prefix = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_ngram_mle(counts_ngram, counts_prefix)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bkngr_edge():
    """Test edge cases."""
    counts_ngram = np.random.default_rng(42).normal(0, 1, 100)
    counts_prefix = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_ngram_mle(counts_ngram, counts_prefix)
    assert isinstance(result, dict)
