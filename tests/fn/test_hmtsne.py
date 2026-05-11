"""Tests for hmtsne.geron_tsne."""
import numpy as np
import pytest
from morie.fn.hmtsne import geron_tsne


def test_hmtsne_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    perplexity = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_tsne(X, n_components, perplexity, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmtsne_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    perplexity = np.random.default_rng(42).normal(0, 1, 100)
    seed = 42
    result = geron_tsne(X, n_components, perplexity, seed)
    assert isinstance(result, dict)
