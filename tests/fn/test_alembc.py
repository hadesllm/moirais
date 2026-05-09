"""Tests for alembc.alammar_embedding_classifier."""
import numpy as np
import pytest
from moirais.fn.alembc import alammar_embedding_classifier


def test_alembc_basic():
    """Test basic functionality."""
    embeddings = np.random.default_rng(42).normal(0, 1, 100)
    labels = np.random.default_rng(43).integers(0, 2, 100)
    classifier = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_embedding_classifier(embeddings, labels, classifier)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alembc_edge():
    """Test edge cases."""
    embeddings = np.random.default_rng(42).normal(0, 1, 100)
    labels = np.random.default_rng(43).integers(0, 2, 100)
    classifier = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_embedding_classifier(embeddings, labels, classifier)
    assert isinstance(result, dict)
