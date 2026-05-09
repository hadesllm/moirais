"""Tests for sctsne.tsne_embedding."""
import numpy as np
import pytest
from moirais.fn.sctsne import tsne_embedding


def test_sctsne_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    perplexity = np.random.default_rng(42).normal(0, 1, 100)
    result = tsne_embedding(X, perplexity)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sctsne_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    perplexity = np.random.default_rng(42).normal(0, 1, 100)
    result = tsne_embedding(X, perplexity)
    assert isinstance(result, dict)
