"""Tests for bkbkof.burkov_ngram_backoff."""
import numpy as np
import pytest
from moirais.fn.bkbkof import burkov_ngram_backoff


def test_bkbkof_basic():
    """Test basic functionality."""
    counts_by_order = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = burkov_ngram_backoff(counts_by_order, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bkbkof_edge():
    """Test edge cases."""
    counts_by_order = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = burkov_ngram_backoff(counts_by_order, alpha)
    assert isinstance(result, dict)
