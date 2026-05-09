"""Tests for albtm.alammar_bertopic_pipeline."""
import numpy as np
import pytest
from moirais.fn.albtm import alammar_bertopic_pipeline


def test_albtm_basic():
    """Test basic functionality."""
    documents = np.random.default_rng(42).normal(0, 1, 100)
    embedder = np.random.default_rng(42).normal(0, 1, 100)
    reducer = np.random.default_rng(42).normal(0, 1, 100)
    cluster_model = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_bertopic_pipeline(documents, embedder, reducer, cluster_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_albtm_edge():
    """Test edge cases."""
    documents = np.random.default_rng(42).normal(0, 1, 100)
    embedder = np.random.default_rng(42).normal(0, 1, 100)
    reducer = np.random.default_rng(42).normal(0, 1, 100)
    cluster_model = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_bertopic_pipeline(documents, embedder, reducer, cluster_model)
    assert isinstance(result, dict)
