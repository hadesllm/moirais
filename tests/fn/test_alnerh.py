"""Tests for alnerh.alammar_ner_token_head."""
import numpy as np
import pytest
from moirais.fn.alnerh import alammar_ner_token_head


def test_alnerh_basic():
    """Test basic functionality."""
    h_tokens = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    tags = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_ner_token_head(h_tokens, W, b, tags)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alnerh_edge():
    """Test edge cases."""
    h_tokens = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    tags = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_ner_token_head(h_tokens, W, b, tags)
    assert isinstance(result, dict)
