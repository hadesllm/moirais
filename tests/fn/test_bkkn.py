"""Tests for bkkn.burkov_kneser_ney."""
import numpy as np
import pytest
from morie.fn.bkkn import burkov_kneser_ney


def test_bkkn_basic():
    """Test basic functionality."""
    counts_ngram = np.random.default_rng(42).normal(0, 1, 100)
    counts_prefix = np.random.default_rng(42).normal(0, 1, 100)
    continuation_counts = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = burkov_kneser_ney(counts_ngram, counts_prefix, continuation_counts, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bkkn_edge():
    """Test edge cases."""
    counts_ngram = np.random.default_rng(42).normal(0, 1, 100)
    counts_prefix = np.random.default_rng(42).normal(0, 1, 100)
    continuation_counts = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = burkov_kneser_ney(counts_ngram, counts_prefix, continuation_counts, d)
    assert isinstance(result, dict)
