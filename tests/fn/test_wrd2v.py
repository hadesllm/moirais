"""Tests for wrd2v.word2vec."""
import numpy as np
import pytest
from moirais.fn.wrd2v import word2vec


def test_wrd2v_basic():
    """Test basic functionality."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = word2vec(corpus, dim, window)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wrd2v_edge():
    """Test edge cases."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = word2vec(corpus, dim, window)
    assert isinstance(result, dict)
