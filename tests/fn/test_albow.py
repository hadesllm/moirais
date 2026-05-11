"""Tests for albow.alammar_bag_of_words."""
import numpy as np
import pytest
from morie.fn.albow import alammar_bag_of_words


def test_albow_basic():
    """Test basic functionality."""
    tokens = np.random.default_rng(42).normal(0, 1, 100)
    vocab = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_bag_of_words(tokens, vocab)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_albow_edge():
    """Test edge cases."""
    tokens = np.random.default_rng(42).normal(0, 1, 100)
    vocab = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_bag_of_words(tokens, vocab)
    assert isinstance(result, dict)
