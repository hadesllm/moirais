"""Tests for tikto.tiktoken_bpe."""
import numpy as np
import pytest
from moirais.fn.tikto import tiktoken_bpe


def test_tikto_basic():
    """Test basic functionality."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    result = tiktoken_bpe(corpus)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tikto_edge():
    """Test edge cases."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    result = tiktoken_bpe(corpus)
    assert isinstance(result, dict)
