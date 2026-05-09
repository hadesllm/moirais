"""Tests for hmpemb.geron_pretrained_embeddings."""
import numpy as np
import pytest
from moirais.fn.hmpemb import geron_pretrained_embeddings


def test_hmpemb_basic():
    """Test basic functionality."""
    vocab = np.random.default_rng(42).normal(0, 1, 100)
    pretrained = np.random.default_rng(42).normal(0, 1, 100)
    freeze = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_pretrained_embeddings(vocab, pretrained, freeze)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmpemb_edge():
    """Test edge cases."""
    vocab = np.random.default_rng(42).normal(0, 1, 100)
    pretrained = np.random.default_rng(42).normal(0, 1, 100)
    freeze = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_pretrained_embeddings(vocab, pretrained, freeze)
    assert isinstance(result, dict)
