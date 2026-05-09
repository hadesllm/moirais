"""Tests for altkemb.alammar_token_embedding_lookup."""
import numpy as np
import pytest
from moirais.fn.altkemb import alammar_token_embedding_lookup


def test_altkemb_basic():
    """Test basic functionality."""
    ids = np.random.default_rng(42).normal(0, 1, 100)
    E_tok = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_token_embedding_lookup(ids, E_tok)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_altkemb_edge():
    """Test edge cases."""
    ids = np.random.default_rng(42).normal(0, 1, 100)
    E_tok = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_token_embedding_lookup(ids, E_tok)
    assert isinstance(result, dict)
