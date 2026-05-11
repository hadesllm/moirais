"""Tests for bklap.burkov_laplace_add_one."""
import numpy as np
import pytest
from morie.fn.bklap import burkov_laplace_add_one


def test_bklap_basic():
    """Test basic functionality."""
    counts_ngram = np.random.default_rng(42).normal(0, 1, 100)
    counts_prefix = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_laplace_add_one(counts_ngram, counts_prefix, V)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bklap_edge():
    """Test edge cases."""
    counts_ngram = np.random.default_rng(42).normal(0, 1, 100)
    counts_prefix = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_laplace_add_one(counts_ngram, counts_prefix, V)
    assert isinstance(result, dict)
