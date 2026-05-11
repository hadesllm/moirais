"""Tests for fastxt.fasttext."""
import numpy as np
import pytest
from morie.fn.fastxt import fasttext


def test_fastxt_basic():
    """Test basic functionality."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = fasttext(corpus, dim)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fastxt_edge():
    """Test edge cases."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = fasttext(corpus, dim)
    assert isinstance(result, dict)
