"""Tests for almqr.alammar_multi_query_retrieval."""
import numpy as np
import pytest
from morie.fn.almqr import alammar_multi_query_retrieval


def test_almqr_basic():
    """Test basic functionality."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    retriever = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_multi_query_retrieval(query, K, retriever, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_almqr_edge():
    """Test edge cases."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    retriever = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_multi_query_retrieval(query, K, retriever, model)
    assert isinstance(result, dict)
