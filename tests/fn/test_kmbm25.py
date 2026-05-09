"""Tests for kmbm25.kamath_bm25_score."""
import numpy as np
import pytest
from moirais.fn.kmbm25 import kamath_bm25_score


def test_kmbm25_basic():
    """Test basic functionality."""
    q_terms = np.random.default_rng(42).normal(0, 1, 100)
    doc_terms = np.random.default_rng(42).normal(0, 1, 100)
    idf = np.random.default_rng(42).normal(0, 1, 100)
    avgdl = np.random.default_rng(42).normal(0, 1, 100)
    k1 = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_bm25_score(q_terms, doc_terms, idf, avgdl, k1, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmbm25_edge():
    """Test edge cases."""
    q_terms = np.random.default_rng(42).normal(0, 1, 100)
    doc_terms = np.random.default_rng(42).normal(0, 1, 100)
    idf = np.random.default_rng(42).normal(0, 1, 100)
    avgdl = np.random.default_rng(42).normal(0, 1, 100)
    k1 = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_bm25_score(q_terms, doc_terms, idf, avgdl, k1, b)
    assert isinstance(result, dict)
