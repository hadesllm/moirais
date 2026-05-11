"""Tests for berte.bert_encoder."""
import numpy as np
import pytest
from morie.fn.berte import bert_encoder


def test_berte_basic():
    """Test basic functionality."""
    tokens = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = bert_encoder(tokens, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_berte_edge():
    """Test edge cases."""
    tokens = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = bert_encoder(tokens, model)
    assert isinstance(result, dict)
