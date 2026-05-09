"""Tests for aldocemb.alammar_document_embedding_pool."""
import numpy as np
import pytest
from moirais.fn.aldocemb import alammar_document_embedding_pool


def test_aldocemb_basic():
    """Test basic functionality."""
    token_embeddings = np.random.default_rng(42).normal(0, 1, 100)
    attention_mask = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_document_embedding_pool(token_embeddings, attention_mask)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aldocemb_edge():
    """Test edge cases."""
    token_embeddings = np.random.default_rng(42).normal(0, 1, 100)
    attention_mask = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_document_embedding_pool(token_embeddings, attention_mask)
    assert isinstance(result, dict)
