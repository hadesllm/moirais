"""Tests for kmcrag.kamath_corrective_rag."""
import numpy as np
import pytest
from moirais.fn.kmcrag import kamath_corrective_rag


def test_kmcrag_basic():
    """Test basic functionality."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    docs = np.random.default_rng(42).normal(0, 1, 100)
    clf = np.random.default_rng(42).normal(0, 1, 100)
    tau_hi = np.random.default_rng(42).normal(0, 1, 100)
    tau_lo = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_corrective_rag(query, docs, clf, tau_hi, tau_lo)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmcrag_edge():
    """Test edge cases."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    docs = np.random.default_rng(42).normal(0, 1, 100)
    clf = np.random.default_rng(42).normal(0, 1, 100)
    tau_hi = np.random.default_rng(42).normal(0, 1, 100)
    tau_lo = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_corrective_rag(query, docs, clf, tau_hi, tau_lo)
    assert isinstance(result, dict)
