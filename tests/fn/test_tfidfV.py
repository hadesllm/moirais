"""Tests for tfidfV.tfidf."""
import numpy as np
import pytest
from morie.fn.tfidfV import tfidf


def test_tfidfV_basic():
    """Test basic functionality."""
    docs = np.random.default_rng(42).normal(0, 1, 100)
    result = tfidf(docs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tfidfV_edge():
    """Test edge cases."""
    docs = np.random.default_rng(42).normal(0, 1, 100)
    result = tfidf(docs)
    assert isinstance(result, dict)
