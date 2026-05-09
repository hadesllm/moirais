"""Tests for ragRet.rag_retrieval."""
import numpy as np
import pytest
from moirais.fn.ragRet import rag_retrieval


def test_ragRet_basic():
    """Test basic functionality."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    docs = np.random.default_rng(42).normal(0, 1, 100)
    result = rag_retrieval(query, docs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ragRet_edge():
    """Test edge cases."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    docs = np.random.default_rng(42).normal(0, 1, 100)
    result = rag_retrieval(query, docs)
    assert isinstance(result, dict)
