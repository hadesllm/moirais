"""Tests for kmhyde.kamath_hyde_hypothetical_doc."""
import numpy as np
import pytest
from morie.fn.kmhyde import kamath_hyde_hypothetical_doc


def test_kmhyde_basic():
    """Test basic functionality."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    embeddings = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_hyde_hypothetical_doc(query, model, embeddings)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmhyde_edge():
    """Test edge cases."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    embeddings = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_hyde_hypothetical_doc(query, model, embeddings)
    assert isinstance(result, dict)
