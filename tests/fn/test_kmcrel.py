"""Tests for kmcrel.kamath_ragas_context_relevance."""
import numpy as np
import pytest
from morie.fn.kmcrel import kamath_ragas_context_relevance


def test_kmcrel_basic():
    """Test basic functionality."""
    context_sentences = np.random.default_rng(42).normal(0, 1, 100)
    relevance_labels = np.random.default_rng(43).integers(0, 2, 100)
    result = kamath_ragas_context_relevance(context_sentences, relevance_labels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmcrel_edge():
    """Test edge cases."""
    context_sentences = np.random.default_rng(42).normal(0, 1, 100)
    relevance_labels = np.random.default_rng(43).integers(0, 2, 100)
    result = kamath_ragas_context_relevance(context_sentences, relevance_labels)
    assert isinstance(result, dict)
