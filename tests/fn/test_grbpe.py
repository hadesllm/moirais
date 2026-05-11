"""Tests for grbpe.geron_bpe_tokenizer_merge."""
import numpy as np
import pytest
from morie.fn.grbpe import geron_bpe_tokenizer_merge


def test_grbpe_basic():
    """Test basic functionality."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    n_merges = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bpe_tokenizer_merge(corpus, n_merges)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grbpe_edge():
    """Test edge cases."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    n_merges = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bpe_tokenizer_merge(corpus, n_merges)
    assert isinstance(result, dict)
