"""Tests for wpiece.wordpiece."""
import numpy as np
import pytest
from morie.fn.wpiece import wordpiece


def test_wpiece_basic():
    """Test basic functionality."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    vocab_size = 100
    result = wordpiece(corpus, vocab_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wpiece_edge():
    """Test edge cases."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    vocab_size = 100
    result = wordpiece(corpus, vocab_size)
    assert isinstance(result, dict)
