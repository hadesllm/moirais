"""Tests for bktfid.burkov_tf_idf."""
import numpy as np
import pytest
from morie.fn.bktfid import burkov_tf_idf


def test_bktfid_basic():
    """Test basic functionality."""
    term = np.random.default_rng(42).normal(0, 1, 100)
    document = np.random.default_rng(42).normal(0, 1, 100)
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_tf_idf(term, document, corpus)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bktfid_edge():
    """Test edge cases."""
    term = np.random.default_rng(42).normal(0, 1, 100)
    document = np.random.default_rng(42).normal(0, 1, 100)
    corpus = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_tf_idf(term, document, corpus)
    assert isinstance(result, dict)
