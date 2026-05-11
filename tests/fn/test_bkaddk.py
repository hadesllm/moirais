"""Tests for bkaddk.burkov_add_k_smoothing."""
import numpy as np
import pytest
from morie.fn.bkaddk import burkov_add_k_smoothing


def test_bkaddk_basic():
    """Test basic functionality."""
    counts_ngram = np.random.default_rng(42).normal(0, 1, 100)
    counts_prefix = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = burkov_add_k_smoothing(counts_ngram, counts_prefix, V, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bkaddk_edge():
    """Test edge cases."""
    counts_ngram = np.random.default_rng(42).normal(0, 1, 100)
    counts_prefix = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = burkov_add_k_smoothing(counts_ngram, counts_prefix, V, k)
    assert isinstance(result, dict)
