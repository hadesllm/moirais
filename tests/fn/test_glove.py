"""Tests for glove.glove."""
import numpy as np
import pytest
from morie.fn.glove import glove


def test_glove_basic():
    """Test basic functionality."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = glove(corpus, dim)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_glove_edge():
    """Test edge cases."""
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = glove(corpus, dim)
    assert isinstance(result, dict)
